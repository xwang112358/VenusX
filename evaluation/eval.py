"""
Standalone evaluation entry point.

Usage — smoke test (no checkpoint needed):

  # token_cls
  python evaluation/eval.py \
      --task token_cls \
      --dataset_name AI4Protein/VenusX_Res_Act_MF50 \
      --plm_type esm \
      --model_name_or_path facebook/esm2_t6_8M_UR50D \
      --max_len 512 --batch_size 4 --num_labels 1 --label_skew \
      --device cpu --smoke

  # fragment_cls
  python evaluation/eval.py \
      --task fragment_cls \
      --dataset_name AI4Protein/VenusX_Frag_Act_MF50 \
      --plm_type esm \
      --model_name_or_path facebook/esm2_t6_8M_UR50D \
      --max_len 512 --batch_size 4 --num_labels 10 \
      --device cpu --smoke
"""

import argparse
import torch
from tqdm import tqdm

from data_loader import build_eval_loader
from smoke_model import SmokeModel
from metrics import build_metrics, report


def create_parser():
    parser = argparse.ArgumentParser(description="VenusX evaluation framework")

    # data
    parser.add_argument('--task', type=str, required=True, choices=['token_cls', 'fragment_cls'])
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="HuggingFace dataset name, e.g. AI4Protein/VenusX_Res_Act_MF50")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=512)

    # encoder (PLM only)
    parser.add_argument('--plm_type', type=str, default='esm',
                        choices=['esm', 'bert', 'ankh', 'saprot', 't5'])
    parser.add_argument('--model_name_or_path', type=str, required=True)

    # output
    parser.add_argument('--num_labels', type=int, default=1,
                        help="1 for token_cls (binary); N classes for fragment_cls")
    parser.add_argument('--label_skew', action='store_true', default=False,
                        help="Use AUPR as primary metric (recommended for token_cls)")

    # runtime
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--smoke', action='store_true', default=False,
                        help="Use SmokeModel (random logits) instead of a real checkpoint")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a .pt checkpoint (unused when --smoke is set)")

    return parser.parse_args()


def load_checkpoint(args):
    """Placeholder — load a real model checkpoint here."""
    raise NotImplementedError(
        "Real checkpoint loading is not yet implemented. "
        "Pass --smoke to run with random logits, or implement this function."
    )


def main():
    args = create_parser()

    print(f"Task        : {args.task}")
    print(f"Dataset     : {args.dataset_name}")
    print(f"PLM         : {args.plm_type} / {args.model_name_or_path}")
    print(f"num_labels  : {args.num_labels}")
    print(f"label_skew  : {args.label_skew}")
    print(f"device      : {args.device}")
    print(f"smoke       : {args.smoke}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading test data...")
    loader = build_eval_loader(args)
    print(f"Test batches: {len(loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.smoke:
        model = SmokeModel(args).to(args.device)
        print("Using SmokeModel (random logits)")
    else:
        model = load_checkpoint(args)
        model = model.to(args.device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    venus_metrics = build_metrics(args.task, args.num_labels, args.label_skew, args.device)

    # ── Eval loop ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            out = model(batch)
            venus_metrics.update(out.logits, batch["target"])

    results = venus_metrics.compute()
    report(results)


if __name__ == '__main__':
    main()
