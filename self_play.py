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
    自動淘汰舊資料的 O(1) 環形連續記憶體緩衝區。
    1. 使用預配置 NumPy 陣列儲存，隨機存取效能極大化。
    2. 容量限制 350,000 筆，避免早期對局污染。
    3. state 以 uint8 存儲，抽樣時才還原 float32，節約 75% 記憶體。
    """

    def __init__(self, max_size=350_000):
        self.max_size = max_size
        self.states_buf = np.zeros((max_size, 4, 15, 15), dtype=np.uint8)
        self.probs_buf = np.zeros((max_size, 225), dtype=np.float32)
        self.values_buf = np.zeros((max_size,), dtype=np.float32)
        self.threats_buf = np.zeros((max_size, 15, 15), dtype=np.float32)
        self.pos = 0
        self.size = 0  # 當前有效樣本數

    def add_game(self, game_data):
        """
        加入一場完整對弈的訓練數據，自動施加 D8 增強。
        """
        for state, probs, value, threat in game_data:
            state_uint8 = state.astype(np.uint8)
            for aug_state, aug_probs, aug_threat in d8_augment(state_uint8, probs, threat):
                self.states_buf[self.pos] = aug_state
                self.probs_buf[self.pos] = aug_probs
                self.values_buf[self.pos] = value
                self.threats_buf[self.pos] = aug_threat
                self.pos = (self.pos + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        """
        隨機抽取一個 mini-batch。使用 NumPy C-Level 索引高速取樣。
        """
        n = min(batch_size, self.size)
        if n == 0:
            return np.zeros((0, 4, 15, 15), dtype=np.float32), np.zeros((0, 225), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0, 15, 15), dtype=np.float32)
        indices = np.random.randint(0, self.size, size=n)
        states = self.states_buf[indices].astype(np.float32)
        probs = self.probs_buf[indices]
        values = self.values_buf[indices]
        threats = self.threats_buf[indices]
        return states, probs, values, threats

    def __len__(self):
        return self.size


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
# 🚀 輕量級自我對弈工作行程 (Self-Play Worker Process)
# ==========================================
def _self_play_worker(worker_id, shm_names, request_queue, worker_event, max_workers,
                      n_games, n_playout, c_puct, temp_threshold, dirichlet_alpha,
                      result_queue):
    """
    自我對弈子處理序：完全不載入 PyTorch/CUDA 函式庫，只透過 CPU 做 MCTS 搜尋
    並透過共享記憶體發送預測請求，徹底消除 WinError 1455 分頁檔不足的問題。
    """
    import signal
    import gc
    import os
    import sys
    import traceback
    
    # 🛡️ 子處理序中忽略 Ctrl+C (SIGINT)，由主處理序統一回收
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # 延遲導入，避免模組循環依賴
    from prediction_server import PredictionClient
    from mcts import LOCAL_WORKER_CACHE

    client = None
    try:
        # 🔥 Numba JIT 預熱，避免第一局搜尋超時
        from env import GomokuEnv as _WarmupEnv
        _w = _WarmupEnv()
        _w.reset()
        _w.step(112)
        _ = _w.get_threat_target()
        _ = _w.get_legal_moves()
        del _w

        client = PredictionClient(
            worker_id=worker_id, shm_names=shm_names, request_queue=request_queue,
            worker_event=worker_event, max_workers=max_workers,
        )
        for _ in range(n_games):
            game_data, winner, moves = play_one_game(
                client.predict_for_mcts,
                n_playout=n_playout, c_puct=c_puct,
                temp_threshold=temp_threshold, dirichlet_alpha=dirichlet_alpha,
            )
            result_queue.put((game_data, winner, moves))
            
            # 🚀 立即釋放 MCTS 快取與記憶體，防止記憶體堆積
            LOCAL_WORKER_CACHE.cache.clear()
            gc.collect()
    except Exception as e:
        crash_msg = f"Worker {worker_id} crashed:\n{traceback.format_exc()}"
        try:
            with open(f"worker_crash_{worker_id}.log", "w", encoding="utf-8") as f:
                f.write(crash_msg)
        except Exception:
            pass
        print(crash_msg, file=sys.stderr, flush=True)
        os._exit(1)
    finally:
        if client is not None:
            client.close()

