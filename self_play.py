"""
self_play.py - Phase 3: 左右互搏戰場 (v7 Pure Zero)
=====================================
AlphaZero 自我對弈引擎。
包含：D8 對稱增強（含 Aux Target）、滑動窗口經驗池、4-tuple 數據、溫度退火。
"""

import numpy as np
from collections import deque
import random

from env import GomokuEnv
from mcts import MCTSEngine

# ==========================================
# 📐 全域常數
# ==========================================
BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE  # 225


# ==========================================
# 🔄 D8 對稱群資料增強 (4 旋轉 × 2 翻轉 = 8 倍)
# ==========================================
def d8_augment(state, mcts_probs, threat_map):
    """
    對棋盤狀態、MCTS 機率分佈、及 Aux 威脅熱力圖施加 D8 對稱群變換。

    Args:
        state: shape (4, 15, 15) 的 NumPy 陣列
        mcts_probs: shape (225,) 的 MCTS 機率向量
        threat_map: shape (15, 15) 的威脅熱力圖 (Aux Target)

    Returns:
        list of (aug_state, aug_probs, aug_threat): 8 組變換結果
    """
    augmented = []
    prob_board = mcts_probs.reshape(BOARD_SIZE, BOARD_SIZE)

    for k in range(4):
        # 旋轉 k × 90 度
        rot_state = np.rot90(state, k, axes=(1, 2)).copy()
        rot_probs = np.rot90(prob_board, k).copy()
        rot_threat = np.rot90(threat_map, k).copy()
        augmented.append((rot_state, rot_probs.flatten(), rot_threat))

        # 旋轉 + 水平翻轉
        flip_state = np.flip(rot_state, axis=2).copy()
        flip_probs = np.flip(rot_probs, axis=1).copy()
        flip_threat = np.flip(rot_threat, axis=1).copy()
        augmented.append((flip_state, flip_probs.flatten(), flip_threat))

    return augmented


# ==========================================
# 🗄️ 滑動窗口經驗池 (Sliding Window Replay Buffer)
# ==========================================
class ReplayBuffer:
    """
    自動淘汰舊資料的 O(1) 環形緩衝區。
    滿載後，最新的高品質對弈數據會自動擠掉最早期的垃圾數據。
    """

    def __init__(self, max_size=200_000):
        self.buffer = deque(maxlen=max_size)

    def add_game(self, game_data):
        """
        加入一場完整對弈的訓練數據，自動施加 D8 增強。

        Args:
            game_data: list of (state, mcts_probs, value, threat_map)
                state: (4, 15, 15) float32
                mcts_probs: (225,) float32
                value: float in [-1, 1]
                threat_map: (15, 15) float32 - Aux Target
        """
        for state, probs, value, threat in game_data:
            for aug_state, aug_probs, aug_threat in d8_augment(state, probs, threat):
                self.buffer.append((
                    aug_state.astype(np.float32),
                    aug_probs.astype(np.float32),
                    np.float32(value),
                    aug_threat.astype(np.float32),
                ))

    def sample_batch(self, batch_size):
        """
        隨機抽取一個 mini-batch。

        Returns:
            (states, probs, values, threats): 各為 NumPy 陣列
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([b[0] for b in batch], dtype=np.float32)
        probs = np.array([b[1] for b in batch], dtype=np.float32)
        values = np.array([b[2] for b in batch], dtype=np.float32)
        threats = np.array([b[3] for b in batch], dtype=np.float32)
        return states, probs, values, threats

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 🎮 單場自我對弈 (Self-Play Game)
# ==========================================
def play_one_game(predict_fn, n_playout=400, c_puct=5.0, temp_threshold=12,
                  dirichlet_alpha=0.3):
    """
    讓 AI 自己跟自己下一盤完整的五子棋。

    Args:
        predict_fn: 接受 (4,15,15) 回傳 ([(action,prob),...], value) 的函式
        n_playout: 每步的 MCTS 模擬次數
        c_puct: 探索/利用平衡常數
        temp_threshold: 前幾步使用高溫探索 (tau=1)
        dirichlet_alpha: Dirichlet 噪聲集中度 (0=無噪聲)

    Returns:
        (training_data, winner, move_count):
            training_data: list of (state, probs_225, value, threat_map)
            winner: 1(黑勝), -1(白勝), 0(平手)
            move_count: 總步數
    """
    env = GomokuEnv()
    env.reset()
    mcts = MCTSEngine(c_puct=c_puct, n_playout=n_playout)

    game_states = []   # (4, 15, 15) 觀測張量
    game_probs = []    # (225,) MCTS 搜尋機率
    game_players = []  # 記錄每一步是誰下的
    game_threats = []  # (15, 15) Aux Target - 威脅熱力圖
    move_count = 0

    while not env.done:
        # 4ch 純粹觀測
        state = env._get_obs_fast()

        # 🧪 收集 Aux Target（只在根節點呼叫一次，不在 MCTS 內部）
        threat_target = env.get_threat_target()

        # 動態溫度退火
        temperature = 1.0 if move_count < temp_threshold else 1e-3

        # MCTS 搜尋
        acts, probs = mcts.get_action_probs(
            env, predict_fn,
            temperature=temperature,
            dirichlet_alpha=dirichlet_alpha,
        )

        # 🛡️ 防禦：MCTS 回傳空結果 (終局邊界)
        if len(acts) == 0:
            break

        # 建構 225 維完整機率向量
        full_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a, p in zip(acts, probs):
            full_probs[a] = p

        # 記錄訓練數據
        game_states.append(state)
        game_probs.append(full_probs)
        game_players.append(env.current_player)
        game_threats.append(threat_target)

        # 選擇落子
        if move_count < temp_threshold:
            action = np.random.choice(acts, p=probs)
        else:
            action = acts[np.argmax(probs)]

        # 執行落子
        env.step(action)

        # 保留子樹
        mcts.update_with_move(action)
        move_count += 1

    # 回溯標記每一步的勝負值
    winner = env.winner
    training_data = []
    for state, probs, player, threat in zip(game_states, game_probs, game_players, game_threats):
        if winner == 0:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        training_data.append((state, probs, value, threat))  # 4-tuple

    return training_data, winner, move_count


# ==========================================
# 🏟️ 自我對弈控制器 (Self-Play Controller)
# ==========================================
class SelfPlayController:
    """
    Phase 3 的總控台。管理自我對弈循環、經驗池、
    並提供訓練數據給 Phase 4 的訓練迴圈。
    """

    def __init__(self, predict_fn, replay_buffer,
                 n_playout=400, c_puct=5.0, temp_threshold=12):
        self.predict_fn = predict_fn
        self.replay_buffer = replay_buffer
        self.n_playout = n_playout
        self.c_puct = c_puct
        self.temp_threshold = temp_threshold

        # 統計
        self.games_played = 0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0

    def play_games(self, n_games=1):
        """
        執行 n_games 場自我對弈，數據自動灌入 Replay Buffer。

        Args:
            n_games: 要下幾盤

        Returns:
            list of (winner, move_count) 每場的結果
        """
        results = []

        for i in range(n_games):
            game_data, winner, move_count = play_one_game(
                self.predict_fn,
                n_playout=self.n_playout,
                c_puct=self.c_puct,
                temp_threshold=self.temp_threshold,
            )

            # D8 增強後灌入經驗池
            self.replay_buffer.add_game(game_data)

            # 更新統計
            self.games_played += 1
            if winner == 1:
                self.black_wins += 1
            elif winner == -1:
                self.white_wins += 1
            else:
                self.draws += 1

            results.append((winner, move_count))
            print(f"  Game {self.games_played}: "
                  f"{'Black' if winner == 1 else 'White' if winner == -1 else 'Draw':>5} wins | "
                  f"{move_count} moves | "
                  f"Buffer: {len(self.replay_buffer):,}")

        return results

    def get_stats(self):
        return {
            'games': self.games_played,
            'black_wins': self.black_wins,
            'white_wins': self.white_wins,
            'draws': self.draws,
            'buffer_size': len(self.replay_buffer),
        }


# ==========================================
# 🧪 Phase 3 整合測試
# ==========================================
if __name__ == '__main__':
    print("=" * 64)
    print("  Phase 3: Self-Play Integration Test (v7 Pure Zero)")
    print("=" * 64)

    # 使用隨機預測器進行快速測試 (不需要 GPU)
    def random_predict_fn(state_tensor):
        """模擬神經網路：均勻機率 + 隨機勝率"""
        probs = np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE
        action_probs = list(enumerate(probs))
        value = np.random.uniform(-0.1, 0.1)
        return action_probs, value

    # 測試 D8 增強 (含 threat_map)
    print("\n[1] D8 Symmetry Augmentation Test (with threat_map)...")
    dummy_state = np.random.randn(4, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
    dummy_probs = np.random.dirichlet(np.ones(ACTION_SIZE)).astype(np.float32)
    dummy_threat = np.random.rand(BOARD_SIZE, BOARD_SIZE).astype(np.float32)
    augmented = d8_augment(dummy_state, dummy_probs, dummy_threat)
    assert len(augmented) == 8, f"Expected 8 augmentations, got {len(augmented)}"
    for i, (s, p, t) in enumerate(augmented):
        assert s.shape == (4, BOARD_SIZE, BOARD_SIZE), f"Aug {i} state shape wrong: {s.shape}"
        assert p.shape == (ACTION_SIZE,), f"Aug {i} probs shape wrong"
        assert t.shape == (BOARD_SIZE, BOARD_SIZE), f"Aug {i} threat shape wrong: {t.shape}"
        assert abs(p.sum() - 1.0) < 1e-5, f"Aug {i} probs don't sum to 1"
    print("  [OK] 8 augmentations with threat_map, all shapes correct")

    # 測試 ReplayBuffer (4-tuple)
    print("\n[2] Replay Buffer Test (4-tuple)...")
    buf = ReplayBuffer(max_size=1000)
    fake_game = [(dummy_state, dummy_probs, 1.0, dummy_threat) for _ in range(5)]
    buf.add_game(fake_game)
    assert len(buf) == 40, f"Expected 40 (5 steps x 8 augs), got {len(buf)}"
    states, probs, values, threats = buf.sample_batch(16)
    assert states.shape == (16, 4, BOARD_SIZE, BOARD_SIZE), f"states shape: {states.shape}"
    assert probs.shape == (16, ACTION_SIZE)
    assert values.shape == (16,)
    assert threats.shape == (16, BOARD_SIZE, BOARD_SIZE), f"threats shape: {threats.shape}"
    print(f"  [OK] Buffer size: {len(buf)}, batch sample shapes correct")

    # 測試自我對弈 (極小模擬次數以加速)
    print("\n[3] Self-Play Game Test (n_playout=8, quick mode)...")
    replay = ReplayBuffer()
    controller = SelfPlayController(
        predict_fn=random_predict_fn,
        replay_buffer=replay,
        n_playout=8,
        temp_threshold=6,
    )
    results = controller.play_games(n_games=2)

    stats = controller.get_stats()
    print(f"\n  Stats: {stats}")
    assert stats['games'] == 2
    assert len(replay) > 0
    print(f"  [OK] 2 games completed, {len(replay):,} samples in buffer")

    print(f"\n{'=' * 64}")
    print(f"  ALL PHASE 3 TESTS PASSED (v7)")
    print(f"{'=' * 64}")
