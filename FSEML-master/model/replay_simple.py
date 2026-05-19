import math
import os

import torch
from torchvision.utils import save_image


class SimpleReplayBuffer:

    mode_name = "simple"

    def __init__(self, args):
        self.enabled = getattr(args, "replay", False)
        self.buffer_size = getattr(args, "replay_buffer_size", 1000)
        self.replay_gap = max(1, getattr(args, "replay_gap", 960))
        self.replay_rate = max(0.0, min(1.0, getattr(args, "replay_rate", 0.05)))
        self.visualize = getattr(args, "visualize_replay", False)
        self.viz_dir = getattr(args, "replay_viz_dir", "replay_visualizations")
        self.viz_max = max(0, getattr(args, "replay_viz_max", 10))

        self.viz_count = 0
        self.latents = []
        self.labels = []
        self.position = 0

    def should_replay(self, meta_iteration):
        if not self.enabled:
            return False
        if not self.latents:
            return False
        return (meta_iteration + 1) % self.replay_gap == 0

    def store_batch(self, model, x, y):
        if not self.enabled or self.buffer_size <= 0:
            return

        with torch.no_grad():
            latent = model.forward_srn(x)["latent"].detach().cpu()
            labels = y.detach().view(-1).cpu()

        for latent_item, label_item in zip(latent, labels):
            latent_item = latent_item.clone()
            label_item = label_item.clone()
            if len(self.latents) < self.buffer_size:
                self.latents.append(latent_item)
                self.labels.append(label_item)
            else:
                self.latents[self.position] = latent_item
                self.labels[self.position] = label_item
                self.position = (self.position + 1) % self.buffer_size

    def sample(self, model, query_batch_size, device, meta_iteration):
        if not self.should_replay(meta_iteration):
            return None, None

        replay_count = int(round(query_batch_size * self.replay_rate))
        replay_count = max(1, replay_count)
        replay_count = min(replay_count, len(self.latents))
        if replay_count <= 0:
            return None, None

        indices = torch.randperm(len(self.latents))[:replay_count].tolist()
        latent_batch = torch.stack([self.latents[idx] for idx in indices]).to(device)
        label_batch = torch.stack([self.labels[idx] for idx in indices]).to(device)

        with torch.no_grad():
            pseudo_samples = model.srn.fssae.decode(latent_batch).detach()

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
            f"replay_step_{meta_iteration + 1:06d}"
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
