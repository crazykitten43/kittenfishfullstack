"""Alpha-beta search harness for the KittenFish L4 SE-ResNet value model.

Run from this directory:
    python search_l4.py --checkpoint checkpoints/kittenfish_l4_resnet.pt --depth 4

The network predicts pawns from the side-to-move perspective. Search converts to
centipawns and uses negamax sign changes. Root children are scored in one batch
to seed move ordering; deeper alpha-beta leaves currently use single-position
inference, so this is a testing harness rather than a fully batch-saturated GPU
search.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import chess

from inference import BatchedEvaluator

MATE = 100_000
INF = 1_000_000
MATE_THRESHOLD = MATE - 1_000
PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000,
}


class SearchTimeout(Exception):
    pass


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: str  # EXACT, LOWER, UPPER
    best_move: chess.Move | None


class Searcher:
    def __init__(self, evaluator: BatchedEvaluator, qdepth: int = 8):
        self.evaluator = evaluator
        self.qdepth = qdepth
        self.nodes = 0
        self.qnodes = 0
        self.deadline = 0.0
        self.tt: dict[tuple[str, int], TTEntry] = {}
        self.killers: dict[int, list[chess.Move]] = {}
        self.history: dict[tuple[bool, int, int], int] = {}
        self.root_order_scores: dict[str, int] = {}

    def check_time(self) -> None:
        if (self.nodes & 511) == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout

    @staticmethod
    def terminal_score(board: chess.Board, ply: int) -> int | None:
        if board.is_checkmate():
            return -MATE + ply
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.can_claim_draw()):
            return 0
        return None

    @staticmethod
    def move_key(move: chess.Move) -> str:
        return move.uci()

    @staticmethod
    def capture_score(board: chess.Board, move: chess.Move) -> int:
        if not board.is_capture(move):
            return 0
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
        attacker = board.piece_type_at(move.from_square)
        if victim is None or attacker is None:
            return 0
        return 10_000 + 10 * PIECE_VALUE[victim] - PIECE_VALUE[attacker]

    def ordered_moves(self, board: chess.Board, ply: int,
                      tt_move: chess.Move | None = None) -> list[chess.Move]:
        moves = list(board.legal_moves)
        killers = self.killers.get(ply, [])
        def score(move: chess.Move) -> int:
            if move == tt_move:
                return 2_000_000
            if move.promotion:
                return 1_000_000 + PIECE_VALUE.get(move.promotion, 0)
            tactical = self.capture_score(board, move)
            if tactical:
                return tactical
            if move in killers:
                return 900_000 - killers.index(move) * 100
            return self.history.get((board.turn, move.from_square, move.to_square), 0)
        moves.sort(key=score, reverse=True)
        return moves

    def evaluate_cp(self, board: chess.Board) -> int:
        # BatchedEvaluator's output is pawns from the side-to-move perspective.
        return round(self.evaluator.evaluate(board.fen()) * 100)

    def quiescence(self, board: chess.Board, alpha: int, beta: int,
                   ply: int, qply: int = 0) -> int:
        self.nodes += 1
        self.qnodes += 1
        self.check_time()
        terminal = self.terminal_score(board, ply)
        if terminal is not None:
            return terminal
        in_check = board.is_check()
        if qply >= self.qdepth:
            return self.evaluate_cp(board)
        if not in_check:
            stand_pat = self.evaluate_cp(board)
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            moves = [m for m in board.legal_moves
                     if board.is_capture(m) or m.promotion]
            moves.sort(key=lambda m: self.capture_score(board, m), reverse=True)
        else:
            # Never stand pat while in check; all legal evasions must be considered.
            moves = self.ordered_moves(board, ply)
        for move in moves:
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int,
                ply: int = 0) -> int:
        self.nodes += 1
        self.check_time()
        terminal = self.terminal_score(board, ply)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        original_alpha = alpha
        key = (board.fen(), board.halfmove_clock)
        entry = self.tt.get(key)
        tt_move = entry.best_move if entry else None
        if entry and entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.score
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.score)
            elif entry.flag == "UPPER":
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        best_score = -INF
        best_move = None
        for move in self.ordered_moves(board, ply, tt_move):
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                if not board.is_capture(move) and not move.promotion:
                    killers = self.killers.setdefault(ply, [])
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]
                    hkey = (board.turn, move.from_square, move.to_square)
                    self.history[hkey] = self.history.get(hkey, 0) + depth * depth
                break

        flag = "EXACT"
        if best_score <= original_alpha:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"
        self.tt[key] = TTEntry(depth, best_score, flag, best_move)
        return best_score

    def batch_order_root(self, board: chess.Board) -> list[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return []
        child_fens = []
        for move in moves:
            board.push(move)
            child_fens.append(board.fen())
            board.pop()
        # Values are from each child's side-to-move perspective; negate them to
        # compare candidates from the root player's perspective.
        values = self.evaluator.evaluate_many(child_fens)
        self.root_order_scores = {
            move.uci(): round(-value * 100) for move, value in zip(moves, values)
        }
        moves.sort(key=lambda m: (
            self.root_order_scores.get(m.uci(), 0) +
            (self.capture_score(board, m) if board.is_capture(m) else 0)
        ), reverse=True)
        return moves

    def search(self, board: chess.Board, max_depth: int = 4,
               time_limit: float = 10.0) -> tuple[chess.Move | None, int | None]:
        self.nodes = self.qnodes = 0
        self.tt.clear()
        self.killers.clear()
        self.history.clear()
        self.deadline = time.monotonic() + max(0.01, time_limit)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, self.terminal_score(board, 0)
        # Batched root evaluation gives the neural net useful work in a batch
        # and supplies a model-informed ordering for the alpha-beta iterations.
        root_moves = self.batch_order_root(board)
        best_move, best_score = root_moves[0], None
        start = time.monotonic()

        for depth in range(1, max_depth + 1):
            try:
                alpha = -INF
                iteration_move, iteration_score = None, -INF
                entry = self.tt.get((board.fen(), board.halfmove_clock))
                tt_move = entry.best_move if entry else None
                moves = self.ordered_moves(board, 0, tt_move)
                # Keep batched model ordering as the stable tie-breaker.
                root_rank = {m: i for i, m in enumerate(root_moves)}
                moves.sort(key=lambda m: (
                    2_000_000 if m == tt_move else
                    self.capture_score(board, m) if board.is_capture(m) else
                    -root_rank.get(m, len(root_rank)
                ), reverse=True)
                for move in moves:
                    board.push(move)
                    try:
                        score = -self.negamax(board, depth - 1, -INF, -alpha, 1)
                    finally:
                        board.pop()
                    if score > iteration_score:
                        iteration_score, iteration_move = score, move
                    alpha = max(alpha, score)
                if iteration_move is not None:
                    best_move, best_score = iteration_move, iteration_score
                elapsed = time.monotonic() - start
                print(
                    f"depth={depth} bestmove={best_move.uci()} "
                    f"score={best_score}cp nodes={self.nodes} qnodes={self.qnodes} "
                    f"time={elapsed:.2f}s nps={self.nodes/max(elapsed, 0.001):.0f} "
                    f"tt={len(self.tt)}",
                    flush=True,
                )
            except SearchTimeout:
                break
        return best_move, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/kittenfish_l4_resnet.pt")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Maximum batch size for root-child scoring")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--qdepth", type=int, default=8)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    args = parser.parse_args()
    board = chess.Board(args.fen)
    if not board.is_valid():
        parser.error("FEN does not describe a valid chess position")
    evaluator = BatchedEvaluator(args.checkpoint, device=args.device,
                                 batch_size=args.batch_size)
    searcher = Searcher(evaluator, qdepth=args.qdepth)
    move, score = searcher.search(board, max_depth=args.depth, time_limit=args.time)
    print(f"bestmove {move.uci() if move else '(none)'}")
    if score is not None:
        print(f"score {score} cp (side to move)")
    print(f"nodes {searcher.nodes}; qnodes {searcher.qnodes}")


if __name__ == "__main__":
    main()
