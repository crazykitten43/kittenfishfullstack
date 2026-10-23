"""Basic alpha-beta negamax search prototype for KittenFish."""
import time
import sys
import chess
from evaluator import evaluate

MATE = 100_000
INF = 1_000_000
PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000,
}

class SearchTimeout(Exception):
    pass

class Searcher:
    def __init__(self):
        self.nodes = 0
        self.deadline = None
        self.stop_requested = False

    def check_time(self):
        if self.stop_requested:
            raise SearchTimeout
        if (self.nodes & 1023) == 0 and self.deadline is not None:
            if time.monotonic() >= self.deadline:
                raise SearchTimeout

    @staticmethod
    def terminal_score(board, ply):
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        return None

    @staticmethod
    def move_order_score(board, move):
        if move.promotion:
            return 20_000 + move.promotion
        if board.is_capture(move):
            victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            return 10_000 + 10 * PIECE_VALUE[victim] - PIECE_VALUE[attacker]
        return 0

    def ordered_moves(self, board):
        return sorted(board.legal_moves, key=lambda m: self.move_order_score(board, m), reverse=True)

    def quiescence(self, board, alpha, beta, ply):
        self.nodes += 1
        self.check_time()
        terminal = self.terminal_score(board, ply)
        if terminal is not None:
            return terminal
        if not board.is_check():
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
            moves.sort(key=lambda m: self.move_order_score(board, m), reverse=True)
        else:
            moves = self.ordered_moves(board)
        for move in moves:
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply=0):
        self.nodes += 1
        self.check_time()
        terminal = self.terminal_score(board, ply)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)
        best_score = -INF
        for move in self.ordered_moves(board):
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score

    def search(self, board, max_depth=4, time_limit=5.0):
        start_time = time.monotonic()
        self.nodes = 0
        self.deadline = time.monotonic() + time_limit
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, self.terminal_score(board, 0)
        best_move, best_score = legal_moves[0], None
        for depth in range(1, max_depth + 1):
            try:
                self.check_time()
                alpha = -INF
                iteration_move, iteration_score = None, -INF
                for move in self.ordered_moves(board):
                    board.push(move)
                    try:
                        score = -self.negamax(board, depth - 1, -INF, -alpha, 1)
                    finally:
                        board.pop()
                    if score > iteration_score:
                        iteration_score, iteration_move = score, move
                    alpha = max(alpha, score)
                best_move, best_score = iteration_move, iteration_score
                elapsed = time.monotonic() - start_time
                print(f"depth={depth} move={best_move.uci()} score={best_score} cp nodes={self.nodes} time={elapsed:.2f}s nps={self.nodes / max(elapsed, 0.001):.0f}", file=sys.stderr)
            except SearchTimeout:
                break
        return best_move, best_score

if __name__ == "__main__":
    board = chess.Board()
    engine = Searcher()
    move, score = engine.search(board, max_depth=5, time_limit=20.0)
    print("Best move:", move)
    print("Score:", score, "cp (side to move)")
    print("Nodes searched:", engine.nodes)
