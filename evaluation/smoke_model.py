import torch
import torch.nn as nn


class VenusModelOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.hidden_states = kwargs.get('hidden_states', None)
        self.logits = kwargs.get('logits', None)
        self.loss = kwargs.get('loss', None)


class SmokeModel(nn.Module):
    """
    Dummy model that returns random logits of the correct shape.
    Used to verify the evaluation pipeline end-to-end without a real checkpoint.

    token_cls    : logits shape (B, L)       — one score per residue
    fragment_cls : logits shape (B, num_labels) — one score per class
    """

    def __init__(self, args):
        super().__init__()
        self.task = args.task
        self.num_labels = args.num_labels
        self.device = args.device

    def forward(self, batch):
        input_ids = batch["input_ids"]  # (B, L)
        B, L = input_ids.shape

        if self.task == 'token_cls':
            # one logit per residue position
            logits = torch.randn(B, L, device=self.device)
        elif self.task == 'fragment_cls':
            # one logit per class
            logits = torch.randn(B, self.num_labels, device=self.device)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return VenusModelOutput(logits=logits)
