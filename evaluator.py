"""CNN evaluator for the KittenFish prototype."""
from pathlib import Path
import chess
import numpy as np
import torch
from train import ChessCNN, encode_fen

CHECKPOINT = Path("checkpoints/kittenfish.pt")
DEVICE = torch.device("cpu")
torch.set_num_threads(1)

def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT.resolve()}")
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    if checkpoint.get("target") != "side_to_move_pawns":
        raise ValueError("Unexpected checkpoint target; expected side_to_move_pawns")
    model = ChessCNN(channels=checkpoint.get("channels", 64)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model = load_model()

@torch.inference_mode()
def evaluate(board: chess.Board) -> int:
    """Return centipawns from the side-to-move perspective."""
    if board.is_checkmate():
        return -100_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    features = encode_fen(board.fen())
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
    return round(float(model(x.to(DEVICE)).item()) * 100)

if __name__ == "__main__":
    board = chess.Board()
    print("Starting position:", evaluate(board), "cp (side to move)")
    board.push_san("e4")
    print("After 1.e4:", evaluate(board), "cp (side to move)")
