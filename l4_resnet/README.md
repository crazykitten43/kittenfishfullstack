# KittenFish L4 SE-ResNet v1

A first GPU-oriented value network for KittenFish. It is designed for the existing Stockfish-labelled dataset and batched inference on an NVIDIA L4. It is a **value-only** model; it does not yet contain a policy head or MCTS.

## Architecture

- Input: the existing 18 × 8 × 8 board encoding, unchanged from the repository's `train.py`.
- Stem: 3×3 convolution, 64 channels, batch normalization, ReLU.
- Trunk: 8 squeeze-and-excitation residual blocks, 64 channels; each block has two 3×3 convolutions and an identity skip.
- Value head: flattened board features → 256-unit dense layer → scalar side-to-move evaluation in pawns.
- Training: AdamW, game-grouped 80/20 validation split, MSE objective, MAE in centipawns, ReduceLROnPlateau, early stopping, best-checkpoint saving.
- GPU: CUDA mixed precision (FP16 autocast) by default when CUDA is available; disable with `--amp off`.
- Batch inference: `BatchedEvaluator.evaluate_many(fens)` chunks requests into GPU batches.

This is deliberately a v1 value model. The dataset contains `fen, eval_cp, game_id`; no move labels are required. Policy learning can be added once Stockfish best-move labels are generated.

## Setup (from repository root)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r l4_resnet/requirements.txt
```

## Train on the 100k dataset

Assuming `data/positions_100k.csv.gz` is in the repository root:

```bash
cd l4_resnet
python train_l4.py --data ../data/positions_100k.csv.gz --epochs 50 --batch-size 512 --eval-batch-size 1024 --workers 4 --threads 8
```

When run from the `l4_resnet` directory, the default output is `checkpoints/kittenfish_l4_resnet.pt`. For predictable paths, set the output explicitly:

```bash
python train_l4.py --data ../data/positions_100k.csv.gz --output checkpoints/kittenfish_l4_resnet.pt
```

The checkpoint stores architecture settings and validation metadata so inference reconstructs the exact model. Do not commit datasets or trained checkpoints to Git.

## Evaluate

Use the same seed and game-level split as training:

```bash
python evaluate.py --data ../data/positions_100k.csv.gz --checkpoint checkpoints/kittenfish_l4_resnet.pt
```

## Batched inference

```bash
python - <<'PY'
import chess
from inference import BatchedEvaluator
ev = BatchedEvaluator("checkpoints/kittenfish_l4_resnet.pt", batch_size=256)
print(ev.evaluate(chess.STARTING_FEN))
print(ev.evaluate_many([chess.STARTING_FEN, chess.Board().fen()]))
PY
```

Values are from the side-to-move perspective, in pawns. Multiply by 100 for approximate centipawns. This is a learned approximation, not a substitute for tablebases or a tactical search.

## Batch-first beam search

`beam_search_l4.py` is the experimental batch-oriented search. It is a **heuristic beam search**, not alpha-beta and not MCTS.

- Default inference batch size: 256.
- Default beam width: 8 retained positions **per root move**, so each legal root move remains represented.
- At each ply, candidates from all active root lines are scored in batches. The search retains candidates favoured by the side that just moved, then backs up scores through the retained tree using negamax.
- If a layer cannot be completed before the time limit, the search returns the best move from the last completed layer.
- Because pruning can remove branches, scores are heuristic and do not have alpha-beta's minimax guarantees. A move can be missed by the beam.
- The reported `depth` is the maximum ply reached; some pruned branches can end at shallower depths.

Run from this directory with the trained checkpoint in place:

```bash
python beam_search_l4.py --checkpoint checkpoints/kittenfish_l4_resnet.pt --device cuda --batch-size 256 --beam-width 8 --depth 8 --time 10
```

Try a wider beam with `--beam-width 16`, or reduce it if generating candidate positions becomes the bottleneck. Compare multiple fixed positions and actual games against the previous alpha-beta harness (`search_l4.py`); raw node count alone does not establish playing strength.

## Previous alpha-beta harness

`search_l4.py` remains available as the baseline. It uses iterative deepening, negamax alpha-beta, quiescence search, move ordering and a transposition table. Its deeper leaf evaluations are mostly single-position calls, so it does not keep the GPU as busy as the beam prototype.

## Reproducibility and limitations

- Seed: 42 by default; train/validation split groups by `game_id` to reduce leakage between positions from the same game.
- Evaluation labels are divided by 100 and clipped to ±20 pawns, matching the existing training pipeline.
- Input encoding remains 18 planes for compatibility with the current dataset and baseline.
- The value head has no move-policy output. Search evaluates legal child positions using the value and converts sign consistently with the side-to-move convention.
- Initial defaults are a starting point, not evidence of strength. Compare validation MAE and prediction distribution against the prior CNN/ResNet, then run playing tests.
