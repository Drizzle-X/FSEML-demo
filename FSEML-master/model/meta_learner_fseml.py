import logging

import torch
from torch import optim
from torch.nn import functional as F

from model.fseml_model import FSEMLModel, FSSAEConfig
from model.meta_learner_base import BaseContinualMetaLearner
from model.replay_factory import ReplayFactory

logger = logging.getLogger("experiment")


class MetaLearnerFSEML(BaseContinualMetaLearner):

    branch_name = "FSEML"

    def __init__(self, args):
        super().__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.meta_iteration = 0

        config = FSSAEConfig(
            num_classes=1000,
            encoder_channels=getattr(args, "encoder_channels", 32),
            latent_dim=getattr(args, "latent_dim", 256),
            adapter_dim=getattr(args, "adapter_dim", 512),
            sparsity_target=getattr(args, "sparsity_target", 0.05),
            sparsity_weight=getattr(args, "sparsity_weight", 5e-3),
            reconstruction_weight=getattr(args, "reconstruction_weight", 0.1),
            weight_decay_weight=getattr(args, "weight_decay_weight", 0.0),
            fda_weight=getattr(args, "fda_weight", 0.1),
        )
        self.net = FSEMLModel(config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.replay = ReplayFactory.build(args)

    def reset_classifer(self, class_to_reset):
        weight = self.net.cpn.output.weight
        bias = self.net.cpn.output.bias
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))
        bias.data[class_to_reset] = 0.0

    def inner_update(self, x, fast_weights, y):
        srn_outputs = self.net.forward_srn(x)
        latent = srn_outputs["latent"].detach()

        if fast_weights is None:
            fast_weights = [param for _, param in self.net.cpn_named_parameters()]

        logits = self.net.forward_cpn(latent, fast_weights=fast_weights)
        loss = self.net.classification_loss(logits, y)
        grad = torch.autograd.grad(loss, fast_weights, allow_unused=False)

        return [param - self.update_lr * grad_param for param, grad_param in zip(fast_weights, grad)]

    def augment_query_with_replay(self, x, y):
        return self.replay.augment_query(self.net, x, y, self.meta_iteration)

    def meta_loss(self, x, fast_weights, y):
        outputs = self.net.forward_features(x, fast_weights=fast_weights)
        srn_losses = self.net.srn.loss_terms(
            x=x,
            reconstruction=outputs["reconstruction"],
            latent=outputs["latent"],
            labels=y,
        )
        classification = self.net.classification_loss(outputs["logits"], y)
        total = classification + self.net.config.fda_weight * srn_losses["total"]
        return total, outputs["logits"], srn_losses["total"], classification

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        x_traj, y_traj, x_rand, y_rand = self.maybe_apply_label_patch_augmentation(
            x_traj,
            y_traj,
            x_rand,
            y_rand,
        )

        fast_weights = self.inner_update(x_traj[0], None, y_traj[0])
        for k in range(1, self.update_step):
            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k])

        real_query_x = x_rand[0]
        real_query_y = y_rand[0]
        mixed_query_x, mixed_query_y, replay_count = self.augment_query_with_replay(real_query_x, real_query_y)

        meta_loss, logits, loss_auto, loss_pre = self.meta_loss(mixed_query_x, fast_weights, mixed_query_y)

        with torch.no_grad():
            pred_q = F.softmax(logits[: len(real_query_y)], dim=1).argmax(dim=1)
            classification_accuracy = torch.eq(pred_q, real_query_y).sum().item()

        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

        self.replay.store_batch(self.net, real_query_x.detach(), real_query_y.detach())

        classification_accuracy /= len(real_query_y)
        self.meta_iteration += 1

        return classification_accuracy, meta_loss, loss_auto, loss_pre
