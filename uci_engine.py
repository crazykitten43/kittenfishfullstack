"""UCI protocol adapter for KittenFish.

Run from the repository root with:
    python uci_engine.py

UCI protocol responses go to stdout; search diagnostics go to stderr.
"""
import shlex
import sys
import threading
import time

import chess

from search import Searcher


ENGINE_NAME = "KittenFish"
ENGINE_AUTHOR = "crazykitten43"


class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.searcher = Searcher()
        self.search_thread = None
        self.search_lock = threading.Lock()
        self.best_move = None
        self.searching = False

    @staticmethod
    def send(message):
        print(message, flush=True)

    def stop_search(self):
        """Ask a running search to stop and wait for its bestmove response."""
        if self.search_thread and self.search_thread.is_alive():
            self.searcher.stop_requested = True
            self.search_thread.join()
        self.search_thread = None
        self.searching = False

    def set_position(self, tokens):
        # A new position supersedes any old search.
        self.stop_search()
        if len(tokens) < 2:
            return

        try:
            if tokens[1] == "startpos":
                board = chess.Board()
                move_start = 2
            elif tokens[1] == "fen":
                try:
                    move_start = tokens.index("moves", 2)
                except ValueError:
                    move_start = len(tokens)
                fen = " ".join(tokens[2:move_start])
                board = chess.Board(fen)
            else:
                return

            if move_start < len(tokens) and tokens[move_start] == "moves":
                for move_text in tokens[move_start + 1:]:
                    board.push_uci(move_text)
            self.board = board
        except (ValueError, IndexError) as exc:
            print(f"Invalid position command: {exc}", file=sys.stderr, flush=True)

    @staticmethod
    def choose_time(tokens, board):
        """Return (time limit in seconds, optional depth limit)."""
        values = {}
        flags = {"infinite", "ponder"}
        i = 1
        while i < len(tokens):
            key = tokens[i]
            if key in flags:
                values[key] = True
                i += 1
            elif key in {
                "wtime", "btime", "winc", "binc", "movestogo",
                "movetime", "depth", "nodes",
            } and i + 1 < len(tokens):
                try:
                    values[key] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1

        if "depth" in values:
            return 3600.0, max(1, min(values["depth"], 64))

        if "movetime" in values:
            # Keep a little time in reserve for protocol overhead.
            return max(0.01, values["movetime"] / 1000.0 * 0.90), 64

        if "infinite" in values or "ponder" in values:
            return 3600.0, 64

        white_to_move = board.turn == chess.WHITE
        remaining_ms = values.get("wtime" if white_to_move else "btime")
        increment_ms = values.get("winc" if white_to_move else "binc", 0)
        if remaining_ms is None:
            return 3.0, 64

        remaining = max(0.0, remaining_ms / 1000.0)
        increment = max(0.0, increment_ms / 1000.0)
        moves_to_go = max(1, values.get("movestogo", 30))
        allocation = remaining / moves_to_go + 0.70 * increment
        # Never intentionally spend most of the remaining clock on one move.
        allocation = min(allocation, max(0.01, remaining * 0.08))
        allocation = min(allocation, max(0.01, remaining - 0.05))
        return max(0.01, allocation), 64

    def start_search(self, tokens):
        self.stop_search()
        if not any(self.board.legal_moves):
            self.send("bestmove 0000")
            return

        time_limit, max_depth = self.choose_time(tokens, self.board)
        board_snapshot = self.board.copy(stack=True)
        self.best_move = next(iter(board_snapshot.legal_moves), None)
        self.searcher.stop_requested = False
        self.searching = True

        def worker():
            try:
                move, score = self.searcher.search(
                    board_snapshot,
                    max_depth=max_depth,
                    time_limit=time_limit,
                )
                if move is not None:
                    self.best_move = move
            except Exception as exc:
                print(f"Search error: {exc}", file=sys.stderr, flush=True)
            finally:
                move_text = self.best_move.uci() if self.best_move else "0000"
                self.send(f"bestmove {move_text}")
                self.searching = False

        self.search_thread = threading.Thread(target=worker, daemon=True)
        self.search_thread.start()

    def handle(self, line):
        tokens = shlex.split(line)
        if not tokens:
            return
        command = tokens[0]

        if command == "uci":
            self.send(f"id name {ENGINE_NAME}")
            self.send(f"id author {ENGINE_AUTHOR}")
            self.send("uciok")
        elif command == "isready":
            self.send("readyok")
        elif command == "ucinewgame":
            self.stop_search()
            self.board = chess.Board()
            self.searcher = Searcher()
        elif command == "position":
            self.set_position(tokens)
        elif command == "go":
            self.start_search(tokens)
        elif command == "stop":
            self.stop_search()
        elif command == "ponderhit":
            # Pondering is not implemented; searches still have a safety timeout.
            pass
        elif command == "quit":
            self.stop_search()
            raise EOFError


def main():
    engine = UCIEngine()
    try:
        for line in sys.stdin:
            try:
                engine.handle(line.strip())
            except EOFError:
                break
            except Exception as exc:
                print(f"UCI command error: {exc}", file=sys.stderr, flush=True)
    finally:
        engine.stop_search()


if __name__ == "__main__":
    main()
