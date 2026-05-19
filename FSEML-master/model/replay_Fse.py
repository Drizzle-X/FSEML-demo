import math
import os

import torch
from torchvision.utils import save_image


class FSEReplayBuffer:

    mode_name = "fse"

    def __init__(self, args):
        self.enabled = getattr(args, "replay", False)
        self.buffer_size = max(1, getattr(args, "replay_buffer_size", 1000))
        self.replay_gap = max(1, getattr(args, "replay_gap", 960))
        self.replay_rate = max(0.0, min(1.0, getattr(args, "replay_rate", 0.05)))
        self.top_p = max(1, getattr(args, "replay_top_p", 32))
        self.num_partitions = max(1, getattr(args, "replay_partitions", 10))

        self.visualize = getattr(args, "visualize_replay", False)
        self.viz_dir = getattr(args, "replay_viz_dir", "replay_visualizations")
        self.viz_max = max(0, getattr(args, "replay_viz_max", 10))
        self.viz_count = 0

        self.partition_capacity = max(1, self.buffer_size // self.num_partitions)
        self.partitions = [
            {
                "latents": [],
                "labels": [],
                "position": 0,
            }
            for _ in range(self.num_partitions)
        ]
        self.samples_since_replay = 0

    def _partition_index(self, label):
        return int(label) % self.num_partitions

    def _iter_stored_items(self):
        for partition in self.partitions:
            for latent, label in zip(partition["latents"], partition["labels"]):
                yield latent, label

    def _buffer_length(self):
        return sum(len(partition["latents"]) for partition in self.partitions)

    def should_replay(self, meta_iteration):
        del meta_iteration
        if not self.enabled:
            return False
        if self._buffer_length() == 0:
            return False
        return self.samples_since_replay >= self.replay_gap

    def _importance_scores(self, model, x, y):
        scores = []
        params = [param for param in model.parameters() if param.requires_grad]

        for idx in range(len(y)):
            sample_x = x[idx : idx + 1]
            sample_y = y[idx : idx + 1]
            outputs = model.forward_features(sample_x)
            loss = model.classification_loss(outputs["logits"], sample_y)
            grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)

            score = torch.zeros((), device=sample_x.device)
            for grad in grads:
                if grad is not None:
                    score = score + grad.pow(2).sum()
            scores.append(score.detach())

        return torch.stack(scores)

    def _store_item(self, latent_item, label_item):
        partition = self.partitions[self._partition_index(label_item.item())]
        latent_item = latent_item.clone().cpu()
        label_item = label_item.clone().cpu()

        if len(partition["latents"]) < self.partition_capacity:
            partition["latents"].append(latent_item)
            partition["labels"].append(label_item)
        else:
            position = partition["position"]
            partition["latents"][position] = latent_item
            partition["labels"][position] = label_item
            partition["position"] = (position + 1) % self.partition_capacity

    def store_batch(self, model, x, y):
        if not self.enabled:
            return

        self.samples_since_replay += len(y)
        if len(y) == 0:
            return

        flat_y = y.detach().view(-1)
        importance = self._importance_scores(model, x, flat_y)
        top_k = min(self.top_p, len(flat_y))
        top_scores, top_indices = torch.topk(importance, k=top_k, largest=True, sorted=True)

        if top_scores.sum().item() <= 0:
            probs = torch.full((top_k,), 1.0 / top_k, device=top_scores.device)
        else:
            probs = top_scores / top_scores.sum()

        sample_count = min(top_k, self.partition_capacity)
        selected_relative = torch.multinomial(probs, num_samples=sample_count, replacement=False)
        selected_indices = top_indices[selected_relative]

        with torch.no_grad():
            selected_x = x[selected_indices]
            selected_y = flat_y[selected_indices]
            selected_latents = model.forward_srn(selected_x)["latent"].detach()

        for latent_item, label_item in zip(selected_latents, selected_y):
            self._store_item(latent_item, label_item)

    def sample(self, model, query_batch_size, device, meta_iteration):
        if not self.should_replay(meta_iteration):
            return None, None

        del query_batch_size

        replay_count = int(math.floor(self.replay_rate * self.replay_gap))
        replay_count = max(1, replay_count)

        stored_items = list(self._iter_stored_items())
        replay_count = min(replay_count, len(stored_items))
        if replay_count <= 0:
            return None, None

        indices = torch.randperm(len(stored_items))[:replay_count].tolist()
        latent_batch = torch.stack([stored_items[idx][0] for idx in indices]).to(device)
        label_batch = torch.stack([stored_items[idx][1] for idx in indices]).to(device)

        with torch.no_grad():
            pseudo_samples = model.srn.fssae.decode(latent_batch).detach()

        self.samples_since_replay = 0
        return pseudo_samples, label_batch

    def maybe_save_visualization(self, pseudo_samples, pseudo_labels, meta_iteration):
        if not self.visualize:
            return
        if self.viz_count >= self.viz_max:
            return
        if pseudo_samples is None or len(pseudo_samples) == 0:
            return

        os.makedirs(self.viz_dir, exist_ok=True)
        nrow = max(1, int(math.sqrt(len(pseudo_samples))))
        file_name = (
            f"replay_fse_step_{meta_iteration + 1:06d}"
            f"_n{len(pseudo_samples):02d}"
            f"_labels_{'-'.join(map(str, pseudo_labels.detach().cpu().tolist()))}.png"
        )
        output_path = os.path.join(self.viz_dir, file_name)
        save_image(
            pseudo_samples.detach().cpu(),
            output_path,
            nrow=nrow,
            normalize=True,
            value_range=(-1.0, 1.0),
        )
        self.viz_count += 1

    def augment_query(self, model, x, y, meta_iteration):
        pseudo_x, pseudo_y = self.sample(model, len(y), x.device, meta_iteration)
        if pseudo_x is None:
            return x, y, 0

        self.maybe_save_visualization(pseudo_x, pseudo_y, meta_iteration)
        mixed_x = torch.cat([x, pseudo_x], dim=0)
        mixed_y = torch.cat([y, pseudo_y], dim=0)
        return mixed_x, mixed_y, len(pseudo_y)
