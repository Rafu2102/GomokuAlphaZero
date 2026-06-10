import numpy as np
import gymnasium as gym
from numba import njit

# 共用 Numba 函式（從 board_utils.py 匯入，避免循環 import）
from board_utils import (
    _get_ray, _analyze_ray, _check_win_and_forbidden,
    _generate_forbidden_mask, _mock_play_check_overline_or_double_four,
    _verify_suspect_three
)

# ==========================================
# 🔑 64-bit Zobrist 隨機表 (模組級，只生成一次)
# ==========================================
# 索引: [player_index(0=黑,1=白)][x][y]
_rng = np.random.default_rng(42)  # 固定種子確保跨進程一致
ZOBRIST_TABLE = _rng.integers(
    0, np.iinfo(np.int64).max, size=(2, 15, 15), dtype=np.int64
)


# ==========================================
# 🧪 Aux Task 目標生成器 (不進入觀測！只作為訓練預測目標)
# ==========================================
@njit(fastmath=True, cache=True)
def _generate_threat_heatmap(board, color):
    """
    掃描全盤空位，為指定顏色計算威脅分數。
    不在 MCTS playout 中呼叫，只在 self-play 記錄訓練數據時生成 Aux Head 的預測目標。

    分數定義：
      五連 = 1.0, 雙四 = 0.9, 衝四 = 0.8
      雙三 = 0.7, 活三 = 0.6 + 交叉火力微調
    """
    heatmap = np.zeros((15, 15), dtype=np.float32)
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for i in range(15):
        for j in range(15):
            if board[i, j] != 0:
                continue
            board[i, j] = color  # 模擬落子
            max_score = 0.0
            total_fours = 0
            total_threes = 0
            for dx, dy in directions:
                ray = _get_ray(board, i, j, dx, dy)
                is_five, _, s_threes, fours = _analyze_ray(ray, color)
                if is_five:
                    max_score = 1.0
                    break  # 五連即阻斷
                total_fours += fours
                total_threes += s_threes
            if max_score < 1.0:
                if total_fours >= 2:
                    max_score = 0.9
                elif total_fours == 1:
                    max_score = 0.8
                if total_threes >= 2:
                    max_score = max(max_score, 0.7)
                elif total_threes == 1:
                    max_score = max(max_score, 0.6)
                # 交叉火力微調 (不超過 1.0)
                max_score = min(1.0, max_score + 0.05 * total_threes)
            heatmap[i, j] = max_score
            board[i, j] = 0  # 復原
    return heatmap

# ==========================================
# 🎮 OpenAI Gym 環境封裝
# ==========================================

class GomokuEnv(gym.Env):
    """
    AlphaZero 五子棋環境 (v7 Pure Zero)
    觀測: 純粹 4ch (己方/敵方/最後落子/顏色標記)
    禁手: 透過 get_legal_moves() 的 legal mask 處理，不進入 NN 輸入
    """
    def __init__(self):
        super(GomokuEnv, self).__init__()
        self.board_size = 15
        self.action_space = gym.spaces.Discrete(self.board_size * self.board_size)
        
        # 4ch 純粹觀測: 己方棋子、對手棋子、最後落子、顏色標記
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4, 15, 15), dtype=np.float32)
        
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1 # 1=黑, -1=白
        self.last_move = -1
        self.done = False
        self.winner = 0
        self.zobrist_hash = np.int64(0)  # 64-bit 增量哈希
        self.stone_count = 0  # O(1) 滿盤判斷計數器

    def clone(self):
        new_env = GomokuEnv()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.last_move = self.last_move
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.zobrist_hash = self.zobrist_hash
        new_env.stone_count = self.stone_count
        return new_env

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.last_move = -1
        self.done = False
        self.winner = 0
        self.zobrist_hash = np.int64(0)
        self.stone_count = 0
        return self._get_obs()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {"legal_moves": self.get_legal_moves()}
            
        x, y = action // self.board_size, action % self.board_size
        
        # 防止違規落子
        if self.board[x, y] != 0:
            raise ValueError(f"Invalid move at ({x}, {y})")

        # 1. 執行落子
        self.board[x, y] = self.current_player
        self.last_move = action
        self.stone_count += 1
        
        # 2. 增量 Zobrist Hash (只需一次 XOR，O(1))
        player_idx = 0 if self.current_player == 1 else 1
        self.zobrist_hash ^= ZOBRIST_TABLE[player_idx, x, y]
        
        # 3. 勝負與禁手判定
        is_win, is_forbidden = _check_win_and_forbidden(self.board, x, y, self.current_player)
        
        reward = 0
        if is_forbidden:
            self.done = True
            self.winner = -self.current_player # 對手獲勝
            reward = -1.0 # 踩地雷直接懲罰
        elif is_win:
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        else:
            # 檢查是否平手 (滿盤)，O(1)
            if self.stone_count >= 225:
                self.done = True
                self.winner = 0
            else:
                self.current_player *= -1 # 換人

        return self._get_obs(), reward, self.done, {"legal_moves": self.get_legal_moves()}

    def step_fast(self, action):
        """
        MCTS 專用輕量 step：只更新棋盤狀態，不計算 obs/legal_moves。
        比 step() 快數十倍，因為跳過了兩次 _generate_forbidden_mask 全盤掃描。
        """
        if self.done:
            return

        x, y = action // self.board_size, action % self.board_size

        if self.board[x, y] != 0:
            raise ValueError(f"MCTS illegal move at ({x}, {y})")

        # 落子
        self.board[x, y] = self.current_player
        self.last_move = action
        self.stone_count += 1

        # 增量 Zobrist Hash
        player_idx = 0 if self.current_player == 1 else 1
        self.zobrist_hash ^= ZOBRIST_TABLE[player_idx, x, y]

        # 勝負與禁手判定
        is_win, is_forbidden = _check_win_and_forbidden(
            self.board, x, y, self.current_player
        )

        if is_forbidden:
            self.done = True
            self.winner = -self.current_player
        elif is_win:
            self.done = True
            self.winner = self.current_player
        else:
            if self.stone_count >= 225:
                self.done = True
                self.winner = 0
            else:
                self.current_player *= -1

    def undo_move(self, action, saved_state):
        """
        反轉一次 step_fast() 呼叫，將棋盤恢復至落子前的精確狀態。
        供 MCTS make/undo 模式使用，消除 env.clone() 的 Python 物件建構開銷。

        Args:
            action: 要撤銷的落子 (0-224)
            saved_state: step_fast 前擷取的 (last_move, done, winner, current_player) 四元組
        """
        x, y = action // self.board_size, action % self.board_size
        prev_last_move, prev_done, prev_winner, prev_player = saved_state

        # 反轉 Zobrist Hash (XOR 自逆)
        player_idx = 0 if prev_player == 1 else 1
        self.zobrist_hash ^= ZOBRIST_TABLE[player_idx, x, y]

        # 移除棋子
        self.board[x, y] = 0
        self.stone_count -= 1

        # 恢復狀態
        self.last_move = prev_last_move
        self.done = prev_done
        self.winner = prev_winner
        self.current_player = prev_player

    def _get_obs(self):
        """生成 4ch 觀測矩陣 Tensor（統一入口）"""
        return self._get_obs_fast()

    def _get_obs_fast(self):
        """
        純粹 4ch 觀測張量。MCTS playout / 根節點 / 訓練數據全部用這個。
        Ch0: 我方棋子  Ch1: 對手棋子  Ch2: 最後落子  Ch3: 顏色標記
        成本: ~0.01ms (零禁手掃描、零 heatmap)
        """
        obs = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        
        # Channel 0: 當前玩家棋子
        obs[0] = (self.board == self.current_player).astype(np.float32)
        # Channel 1: 對手玩家棋子
        obs[1] = (self.board == -self.current_player).astype(np.float32)
        
        # Channel 2: 對手最後落子位
        if self.last_move != -1:
            lx, ly = self.last_move // self.board_size, self.last_move % self.board_size
            obs[2][lx, ly] = 1.0
            
        # Channel 3: 當前顏色標記 (1=黑棋, 0=白棋)
        if self.current_player == 1:
            obs[3].fill(1.0)

        return obs

    def get_threat_target(self):
        """
        生成 Aux Head 的訓練目標：當前玩家的威脅熱力圖 (15x15)。
        ⚠️ 只在 self-play 根節點收集訓練數據時呼叫（每步 1 次），
        不進入 MCTS playout、不進入 NN 輸入。
        """
        return _generate_threat_heatmap(self.board.copy(), self.current_player)

    def get_legal_moves(self):
        """取得合法步。黑棋回合會額外剔除(遮蔽)觸發禁手的點。"""
        empty_spots = np.where(self.board.flatten() == 0)[0]
        
        if self.current_player == 1:
            # 使用 Numba 禁手遮罩過濾
            mask = _generate_forbidden_mask(self.board).flatten()
            legal_moves = [move for move in empty_spots if mask[move] == 0]
            return legal_moves
        
        return empty_spots.tolist()

if __name__ == "__main__":
    # 效能與基本機制測試
    env = GomokuEnv()
    obs = env.reset()
    print("環境封裝成功！ 4ch Tensor Shape:", obs.shape)
    assert obs.shape == (4, 15, 15), f"觀測維度錯誤: {obs.shape}"
    print("全盤初始合法步數量:", len(env.get_legal_moves()))
    
    # 測試落子
    env.step(112)  # 中間天元
    print("下子後換白棋，黑棋盤面已轉移。目前合法步數量:", len(env.get_legal_moves()))
    
    # 測試 Aux Target
    env.step(113)
    threat = env.get_threat_target()
    print(f"Aux Threat Target Shape: {threat.shape}, Max: {threat.max():.2f}")
    assert threat.shape == (15, 15), f"Threat target 維度錯誤: {threat.shape}"
    print("env.py 所有測試通過！")
