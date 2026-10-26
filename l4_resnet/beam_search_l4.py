"""Batch-first beam search for the KittenFish L4 value network.

Run from this directory:
    python beam_search_l4.py --checkpoint checkpoints/kittenfish_l4_resnet.pt \
        --device cuda --batch-size 256 --beam-width 8 --depth 8 --time 10

This is heuristic beam search, not MCTS or alpha-beta. Each root move keeps
its own beam so no legal root move disappears merely because another line has
a high neural-network score. At each ply, candidate children across all active
lines are evaluated in batches, then the best beam_width children per root
move are retained according to the side that just moved. Retained-tree values
are backed up with negamax.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import chess

from inference import BatchedEvaluator

MATE = 100_000
MATE_THRESHOLD = MATE - 1_000


@dataclass(eq=False)
class SearchNode:
    board: chess.Board
    root_move: chess.Move
    depth: int
    score_cp: int | None = None  # side-to-move perspective
    terminal: bool = False
    parent: "SearchNode | None" = None
    move: chess.Move | None = None
    children: list["SearchNode"] = field(default_factory=list)


class SearchTimeout(Exception):
    """Raised when the next complete beam layer cannot be finished in time."""


class BeamSearcher:
    def __init__(
        self,
        evaluator: BatchedEvaluator,
        beam_width: int = 8,
        batch_size: int = 256,
    ):
        self.evaluator = evaluator
        self.beam_width = max(1, int(beam_width))
        self.batch_size = max(1, int(batch_size))
        self.nodes = 0
        self.evals = 0
        self.batches = 0
        self.deadline = 0.0
        self.root_color = chess.WHITE

    @staticmethod
    def terminal_score(board: chess.Board, ply: int) -> int | None:
        """Return a score from the side-to-move perspective, or None."""
        if board.is_checkmate():
            return -MATE + ply
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            return 0
        return None

    @staticmethod
    def tactical_tiebreak(board: chess.Board, move: chess.Move) -> int:
        """Only a tie-breaker; the neural evaluation remains the main signal."""
        score = 0
        if move.promotion:
            score += 2_000 + move.promotion
        if board.is_capture(move):
            victim = (
                chess.PAWN
                if board.is_en_passant(move)
                else board.piece_type_at(move.to_square)
            )
            attacker = board.piece_type_at(move.from_square)
            values = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 20_000,
            }
            if victim is not None:
                score += 1_000 + 10 * values[victim] - values.get(attacker, 0)
        if board.gives_check(move):
            score += 50
        return score

    def check_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            raise SearchTimeout

    def evaluate_nodes(self, nodes: list[SearchNode]) -> None:
        """Evaluate non-terminal nodes, letting BatchedEvaluator chunk requests."""
        pending: list[SearchNode] = []
        fens: list[str] = []
        for node in nodes:
            terminal_score = self.terminal_score(node.board, node.depth)
            if terminal_score is not None:
                node.score_cp = terminal_score
                node.terminal = True
            else:
                pending.append(node)
                fens.append(node.board.fen())

        if not fens:
            return

        self.evals += len(fens)
        self.batches += math.ceil(len(fens) / self.batch_size)
        values = self.evaluator.evaluate_many(fens)
        for node, value in zip(pending, values):
            node.score_cp = round(value * 100)

    def back_up_values(self, nodes: list[SearchNode]) -> None:
        """Negamax backup over the retained beam tree."""
        for node in reversed(nodes):
            if node.children:
                # Child scores use the child's side-to-move perspective.
                # Negate them, then choose the best retained child for this mover.
                node.score_cp = max(
                    -int(child.score_cp) for child in node.children
                    if child.score_cp is not None
                )

    def root_result(
        self, root_nodes: list[SearchNode]
    ) -> tuple[chess.Move | None, int | None]:
        if not root_nodes:
            return None, None
        # Root-child scores are from the opponent's perspective.
        best = max(root_nodes, key=lambda n: -int(n.score_cp or 0))
        return best.root_move, -int(best.score_cp or 0)

    def search(
        self,
        board: chess.Board,
        max_depth: int = 8,
        time_limit: float = 10.0,
    ) -> tuple[chess.Move | None, int | None]:
        self.nodes = self.evals = self.batches = 0
        self.deadline = time.monotonic() + max(0.01, time_limit)
        self.root_color = board.turn
        start = time.monotonic()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, self.terminal_score(board, 0)

        # Keep every root move alive; beam_width is per root move, not global.
        root_nodes: list[SearchNode] = []
        for move in legal_moves:
            child_board = board.copy(stack=True)
            child_board.push(move)
            root_nodes.append(
                SearchNode(
                    board=child_board,
                    root_move=move,
                    move=move,
                    depth=1,
                )
            )

        self.nodes += len(root_nodes)
        self.evaluate_nodes(root_nodes)
        all_nodes = list(root_nodes)
        frontier = list(root_nodes)
        self.back_up_values(all_nodes)

        best_move, best_score = self.root_result(root_nodes)
        self.print_progress(1, best_move, best_score, start)

        # A beam layer is committed only after its candidates have been
        # generated, evaluated and selected. If time expires mid-layer, the
        # last completed result remains available.
        for ply in range(2, max(1, max_depth) + 1):
            try:
                self.check_deadline()
                candidates_by_root: dict[str, list[SearchNode]] = {}
                generated = 0

                for parent in frontier:
                    if parent.terminal:
                        continue
                    self.check_deadline()
                    for move in parent.board.legal_moves:
                        generated += 1
                        if generated % 128 == 0:
                            self.check_deadline()
                        child_board = parent.board.copy(stack=True)
                        child_board.push(move)
                        candidate = SearchNode(
                            board=child_board,
                            root_move=parent.root_move,
                            parent=parent,
                            move=move,
                            depth=ply,
                        )
                        candidates_by_root.setdefault(parent.root_move.uci(), []).append(
                            candidate
                        )

                candidates = [
                    candidate
                    for group in candidates_by_root.values()
                    for candidate in group
                ]
                if not candidates:
                    break
                self.check_deadline()
                self.nodes += len(candidates)
                self.evaluate_nodes(candidates)

                selected: list[SearchNode] = []
                # All parents in this layer have the same side to move. Rank
                # child states by -child_eval: the mover wants the best score
                # from their own perspective.
                root_lookup = {move.uci(): move for move in legal_moves}
                for root_key, group in candidates_by_root.items():
                    group.sort(
                        key=lambda child: (
                            -int(child.score_cp or 0),
                            self.tactical_tiebreak(child.parent.board, child.move)
                            if child.parent is not None and child.move is not None
                            else 0,
                        ),
                        reverse=True,
                    )
                    kept = group[: self.beam_width]
                    for child in kept:
                        assert child.parent is not None
                        child.parent.children.append(child)
                    selected.extend(kept)

                if not selected:
                    break

                all_nodes.extend(selected)
                frontier = selected
                self.back_up_values(all_nodes)
                best_move, best_score = self.root_result(root_nodes)
                self.print_progress(ply, best_move, best_score, start)

            except SearchTimeout:
                break

        return best_move, best_score

    def print_progress(
        self,
        depth: int,
        best_move: chess.Move | None,
        score: int | None,
        start: float,
    ) -> None:
        elapsed = max(time.monotonic() - start, 0.001)
        move_text = best_move.uci() if best_move else "(none)"
        score_text = f"{score}cp" if score is not None else "n/a"
        print(
            f"depth={depth} bestmove={move_text} score={score_text} "
            f"nodes={self.nodes} evals={self.evals} batches={self.batches} "
            f"time={elapsed:.2f}s nps={self.nodes / elapsed:.0f} "
            f"beam={self.beam_width}/root-move",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="checkpoints/kittenfish_l4_resnet.pt"
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Maximum number of FENs sent to the network in one inference batch",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=8,
        help="Maximum retained positions per root move at each ply",
    )
    parser.add_argument("--depth", type=int, default=8, help="Maximum ply depth")
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    args = parser.parse_args()

    board = chess.Board(args.fen)
    if not board.is_valid():
        parser.error("FEN does not describe a valid chess position")

    evaluator = BatchedEvaluator(
        args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    searcher = BeamSearcher(
        evaluator,
        beam_width=args.beam_width,
        batch_size=args.batch_size,
    )
    move, score = searcher.search(
        board,
        max_depth=args.depth,
        time_limit=args.time,
    )
    print(f"bestmove {move.uci() if move else '(none)'}")
    if score is not None:
        print(f"score {score} cp (side to move)")
    print(
        f"nodes {searcher.nodes}; neural evaluations {searcher.evals}; "
        f"inference batches {searcher.batches}"
    )


if __name__ == "__main__":
    main()
