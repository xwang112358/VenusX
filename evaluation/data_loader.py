import ast
import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    AutoTokenizer,
    EsmTokenizer,
    T5Tokenizer,
)


# ── Dataset ──────────────────────────────────────────────────────────────────

class VenusDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index]


# ── HuggingFace loading ───────────────────────────────────────────────────────

def load_data(args, split='test'):
    """Load a single split from HuggingFace Hub. Returns a list of dicts."""
    split_mapping = {'train': 'train', 'valid': 'validation', 'test': 'test'}
    dataset = load_dataset(args.dataset_name, split=split_mapping[split])

    if args.task == 'fragment_cls':
        return [
            {
                'name': item['uid'],
                'interpro': item['interpro_id'],
                'fragment': item['seq_fragment'],
                'interpro_label': item['interpro_label'],
            }
            for item in dataset
        ]

    elif args.task == 'token_cls':
        return [
            {
                'name': item['uid'],
                'sequence': item['seq_full'],
                'label': ast.literal_eval(item['label']),
            }
            for item in dataset
        ]

    else:
        raise ValueError(f"Unknown task: {args.task}")


# ── PLM collate functions ─────────────────────────────────────────────────────

def _make_tokenizer(args):
    if args.plm_type == 'bert':
        return BertTokenizer.from_pretrained(args.model_name_or_path)
    elif args.plm_type in ['esm', 'ankh']:
        return AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.plm_type == 'saprot':
        return EsmTokenizer.from_pretrained(args.model_name_or_path)
    elif args.plm_type == 't5':
        return T5Tokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    else:
        raise ValueError(f"Unsupported PLM type: {args.plm_type}")


class TokenClsCollateFnForPLM:

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tokenizer = _make_tokenizer(args)

    def __call__(self, batch):
        sequences = [protein["sequence"] for protein in batch]
        labels = [[-100] + protein["label"] + [-100] for protein in batch]

        if self.args.plm_type == 'saprot':
            max_len = max(len(seq) // 2 for seq in sequences)
        else:
            max_len = max(len(seq) for seq in sequences)

        if max_len > self.args.max_len:
            max_len = self.args.max_len

        if self.args.plm_type == 'bert':
            sequences = [" ".join(seq) for seq in sequences]
        if self.args.plm_type == 't5':
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

        if self.args.plm_type == 't5':
            results = self.tokenizer(sequences, add_special_tokens=True, padding="longest")
        else:
            results = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                max_length=max_len,
                truncation=True,
            )

        labels_padded = [
            label[:max_len] + [-100] * (max_len - len(label))
            if len(label) < max_len else label[:max_len]
            for label in labels
        ]
        results["target"] = torch.tensor(labels_padded, dtype=torch.long).to(self.device)
        return results


class FragmentClsCollateFnForPLM:

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tokenizer = _make_tokenizer(args)

    def __call__(self, batch):
        sequences = [protein["fragment"] for protein in batch]
        labels = [protein["interpro_label"] for protein in batch]
        max_len = max(len(seq) for seq in sequences)

        if max_len > self.args.max_len:
            max_len = self.args.max_len

        if self.args.plm_type == 'bert':
            sequences = [" ".join(seq) for seq in sequences]
        if self.args.plm_type == 't5':
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

        if self.args.plm_type == 't5':
            results = self.tokenizer(sequences, add_special_tokens=True, padding="longest")
        else:
            results = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                max_length=max_len,
                truncation=True,
            )

        results["target"] = torch.tensor(labels, dtype=torch.long).to(self.device)
        return results


# ── Public entry point ────────────────────────────────────────────────────────

def build_eval_loader(args):
    """Return a DataLoader over the test split."""
    data = load_data(args, split='test')
    dataset = VenusDataset(data)

    if args.task == 'token_cls':
        collate_fn = TokenClsCollateFnForPLM(args)
    elif args.task == 'fragment_cls':
        collate_fn = FragmentClsCollateFnForPLM(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
    )
