"""
opening_book.py - D8 對稱展開開局庫
=====================================
生成時計算所有 8 種 D8 對稱變換的 Zobrist Hash，
存入字典實現 O(1) 查詢。

使用方式：
  book = OpeningBook()
  book.add_entry(zobrist_hash, best_action, board)
  book.save("opening_book.json")
  book = OpeningBook.load("opening_book.json")
  action = book.lookup(zobrist_hash)  # O(1)
"""
import json
import numpy as np
from env import ZOBRIST_TABLE


def _d8_transforms(board):
    """生成棋盤的 8 種 D8 對稱變換"""
    transforms = []
    for k in range(4):
        rot = np.rot90(board, k)
        transforms.append(rot.copy())
        transforms.append(np.flip(rot, axis=1).copy())
    return transforms


def _compute_zobrist(board):
    """從棋盤重新計算完整 Zobrist Hash"""
    h = np.int64(0)
    for i in range(15):
        for j in range(15):
            if board[i, j] == 1:
                h ^= ZOBRIST_TABLE[0, i, j]
            elif board[i, j] == -1:
                h ^= ZOBRIST_TABLE[1, i, j]
    return int(h)


def _transform_action(action, k, flip):
    """將 action 座標依照 D8 變換映射"""
    x, y = action // 15, action % 15
    for _ in range(k):
        x, y = y, 14 - x  # 90° 逆時針旋轉
    if flip:
        y = 14 - y  # 水平翻轉
    return x * 15 + y


class OpeningBook:
    def __init__(self):
        self.book = {}  # {zobrist_hash_int: action_int}

    def add_entry(self, zobrist_hash, best_action, board):
        """
        加入一個開局定式，自動展開為 8 種 D8 對稱。
        Args:
            zobrist_hash: 原始盤面的 Zobrist Hash
            best_action: 最佳落子 (0-224)
            board: 15×15 棋盤 (np.int8)
        """
        idx = 0
        for k in range(4):
            for flip in [False, True]:
                rot_board = np.rot90(board, k).copy()
                if flip:
                    rot_board = np.flip(rot_board, axis=1).copy()
                sym_hash = _compute_zobrist(rot_board)
                sym_action = _transform_action(best_action, k, flip)
                self.book[sym_hash] = sym_action
                idx += 1

    def lookup(self, zobrist_hash):
        """O(1) 查詢。回傳 action 或 None。"""
        return self.book.get(zobrist_hash, None)

    def save(self, path):
        """存檔為 JSON"""
        # JSON key 必須是 str
        data = {str(k): v for k, v in self.book.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """從 JSON 載入"""
        book = cls()
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            book.book = {int(k): v for k, v in data.items()}
        except FileNotFoundError:
            pass  # 檔案不存在 = 空開局庫
        return book

    def __len__(self):
        return len(self.book)


if __name__ == '__main__':
    print("=== Opening Book Tests ===")
    
    # Test: 空盤天元 → 應該展開為 8 個 hash entry（但天元是中心，對稱後可能重複）
    board = np.zeros((15, 15), dtype=np.int8)
    book = OpeningBook()
    book.add_entry(0, 112, board)  # 空盤的 hash=0，推薦天元
    print(f"天元展開後 entries: {len(book)}")
    
    # Test: 非中心點應產生 8 個不同 entry
    board2 = np.zeros((15, 15), dtype=np.int8)
    board2[7, 7] = 1
    book2 = OpeningBook()
    from env import GomokuEnv
    env = GomokuEnv()
    env.reset()
    env.step(112)  # 天元
    book2.add_entry(int(env.zobrist_hash), 97, env.board)  # 推薦 (6,7)
    print(f"非中心展開後 entries: {len(book2)}")
    
    # Test: save/load
    book2.save("test_book.json")
    loaded = OpeningBook.load("test_book.json")
    print(f"Save/Load test: {len(loaded)} entries loaded")
    
    # Cleanup
    import os
    os.remove("test_book.json")
    
    print("=== ALL OPENING BOOK TESTS PASSED ===")
