"""Train the keyword-spotting MLP and convert it to a spiking NIR graph.

Usage (from the repository root, in an environment with the `snntorch` extra):

    python zoo/kws/train.py --data /path/to/speech_commands_v0.02 --out build/kws

Writes to --out: features cache, `model.pt` (ANN weights), `kws.nir` (spiking graph),
`test.npz` (normalised test features and labels) and `train_metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import KEYWORDS, load_split


def cached_split(data: Path, out: Path, split: str):
    path = out / f"features_{split}.npz"
    if not path.exists():
        x, y = load_split(data, split)
        np.savez_compressed(path, x=x, y=y)
    d = np.load(path)
    return d["x"], d["y"]


def build_model(n_in: int, hidden: int, n_out: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, n_out),
    )


def main() -> None:
    import nir
    import torch

    from pygri.convert import convert_sequential

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("build/kws"))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    splits = {s: cached_split(args.data, args.out, s) for s in ("train", "validation", "test")}
    mean = splits["train"][0].mean(axis=0)
    std = splits["train"][0].std(axis=0) + 1e-6
    t = {
        s: (torch.tensor((x - mean) / std, dtype=torch.float32), torch.tensor(y))
        for s, (x, y) in splits.items()
    }
    x_tr, y_tr = t["train"]

    model = build_model(x_tr.shape[1], args.hidden, len(KEYWORDS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def accuracy(split: str) -> float:
        x, y = t[split]
        with torch.no_grad():
            return float((model(x).argmax(1) == y).float().mean())

    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(perm), 128):
            idx = perm[i : i + 128]
            xb = x_tr[idx] + 0.3 * torch.randn_like(x_tr[idx])  # input noise as regulariser
            opt.zero_grad()
            loss_fn(model(xb), y_tr[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        val = accuracy("validation")
        if val > best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:3d}  val {val:.4f}")
    model.load_state_dict(best_state)

    metrics = {s: accuracy(s) for s in ("train", "validation", "test")}
    print(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), args.out / "model.pt")
    np.savez(args.out / "norm.npz", mean=mean, std=std)

    calib = x_tr[torch.randperm(len(y_tr))[:2000]]
    graph = convert_sequential(model, calib).graph
    nir.write(str(args.out / "kws.nir"), graph)
    np.savez_compressed(args.out / "test.npz", x=t["test"][0].numpy(), y=t["test"][1].numpy())
    (args.out / "train_metrics.json").write_text(
        json.dumps({"ann_accuracy": metrics, "hidden": args.hidden, "epochs": args.epochs}, indent=2)
    )


if __name__ == "__main__":
    main()
