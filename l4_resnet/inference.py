"""Batched inference helpers. Values are in pawns from the side-to-move perspective."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from encoding import encode_fen
from model import KittenFishResNet

class BatchedEvaluator:
    """GPU/CPU evaluator that scores many FENs in one forward pass."""
    def __init__(self, checkpoint: str, device: str = "auto", batch_size: int = 256, amp: bool = True):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = KittenFishResNet(
            input_planes=ckpt.get("input_planes", 18),
            channels=ckpt.get("channels", 64), blocks=ckpt.get("blocks", 8)
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.batch_size = max(1, int(batch_size))
        self.amp = bool(amp and self.device.type == "cuda")
    @torch.inference_mode()
    def evaluate_many(self, fens: Sequence[str]) -> list[float]:
        """Return one side-to-move evaluation in pawns for each input FEN."""
        if not fens: return []
        values = []
        for start in range(0, len(fens), self.batch_size):
            batch = np.stack([encode_fen(f) for f in fens[start:start+self.batch_size]])
            x = torch.from_numpy(batch).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                pred = self.model(x).float()
            values.extend(pred.cpu().tolist())
        return values
    def evaluate(self, fen: str) -> float:
        return self.evaluate_many([fen])[0]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/kittenfish_l4_resnet.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("fens", nargs="*", help="FEN strings; quote each FEN")
    a = p.parse_args()
    evaluator = BatchedEvaluator(a.checkpoint, device=a.device, batch_size=a.batch_size)
    for fen, score in zip(a.fens, evaluator.evaluate_many(a.fens)):
        print(f"{score:+.3f} pawns  {fen}")
