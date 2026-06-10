"""
mcts.py - v7 Pure Zero MCTS 搜尋引擎 (C++ Accelerated)
======================================================
包含：MCTS 節點、Zobrist 轉置表、VCF Root-only Candidate Filter、
      Additive Prior Bias、Proof Propagation、Opening Book 支援。
"""
import numpy as np
import math
import random
from collections import OrderedDict

# ==========================================
# 🔮 Zobrist 轉置表全域快取 (LRU-Bounded Transposition Table)
# ==========================================
class ZobristCache:
    """存放神經網路推論結果，遇到「殊途同歸」的盤面直接提取，節省 GPU 推論。
    使用 OrderedDict 實現 LRU 淘汰，將大小限制在 100,000 筆以防止記憶體洩漏。"""
    def __init__(self, max_size=100_000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, state_bytes):
        result = self.cache.get(state_bytes, None)
        if result is not None:
            self.cache.move_to_end(state_bytes)  # LRU: 移至尾部
        return result
        
    def save(self, state_bytes, policy, value):
        if state_bytes in self.cache:
            self.cache.move_to_end(state_bytes)
        self.cache[state_bytes] = (policy, value)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # 淘汰最舊的 entry

# 在多進程 Async Self-Play 中，每個 Worker 擁有獨立的快取副本。
LOCAL_WORKER_CACHE = ZobristCache()

import mcts_core

# ==========================================
# 🧠 動態 MCTS 搜尋引擎 (C++ Accelerated)
# ==========================================
class MCTSEngine:
    def __init__(self, c_puct=5.0, n_playout=800, opening_book=None):
        self.cpp_tree = mcts_core.CppMCTSTree(c_puct)
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.opening_book = opening_book  # 開局庫實例 (None = 不使用)

    def playout(self, env, predict_fn):
        """執行【單次】腦內模擬：Select -> Expand -> Backup。
        使用 make/undo 模式：過程中修改 env，結束後完整恢復。不碰 VCF。"""
        actions = self.cpp_tree.select()
        
        path_saved = []
        for a in actions:
            saved = (env.last_move, env.done, env.winner, env.current_player)
            env.step_fast(a)
            path_saved.append((a, saved))
            
        if not env.done:
            state_tensor = env._get_obs_fast()
            cache_key = (int(env.zobrist_hash), env.current_player, env.last_move)
            cached_result = LOCAL_WORKER_CACHE.get(cache_key)
            
            if cached_result is not None:
                action_probs, leaf_value = cached_result
            else:
                action_probs, leaf_value = predict_fn(state_tensor)
                legal_moves = env.get_legal_moves()
                action_probs = [(a, p) for a, p in action_probs if a in legal_moves]
                total_p = sum([p for a, p in action_probs])
                if total_p > 0:
                    action_probs = [(a, p / total_p) for a, p in action_probs]
                elif len(action_probs) > 0:
                    prob = 1.0 / len(action_probs)
                    action_probs = [(a, prob) for a, _ in action_probs]
                LOCAL_WORKER_CACHE.save(cache_key, action_probs, leaf_value)
                
            self.cpp_tree.expand(action_probs)
            self.cpp_tree.backup(leaf_value, False, 0)
        else:
            # MCTS Solver (終局證明)
            winner_perspective = 0
            if env.winner == env.current_player:
                winner_perspective = 1
            elif env.winner == -env.current_player:
                winner_perspective = -1
            self.cpp_tree.backup(0.0, True, winner_perspective)
            
        for action, saved in reversed(path_saved):
            env.undo_move(action, saved)

    def get_action_probs(self, env, predict_fn, temperature=1e-3, dirichlet_alpha=0.3):
        """
        根據 MCTS 探索次數 (N)，結合 VCF 候選過濾與 Dirichlet Noise，輸出落子機率分佈。
        VCF 只在此處根節點跑 1 次，不在 playout() 內部。
        """
        # 📚 開局庫查詢（前 12 手）
        if self.opening_book is not None:
            stone_count = np.count_nonzero(env.board)
            if stone_count < 12:
                book_action = self.opening_book.lookup(int(env.zobrist_hash))
                if book_action is not None:
                    probs = np.zeros(225)
                    probs[book_action] = 1.0
                    return list(range(225)), probs

        stone_count = np.count_nonzero(env.board)
        vcf_threat_actions = []
        if stone_count >= 20:
            from vcf_vct import vcf_search, _find_four_threats
            vcf_win = vcf_search(env.board.copy(), env.current_player, max_depth=15)
            if vcf_win:
                threats = _find_four_threats(env.board.copy(), env.current_player)
                for t in range(len(threats)):
                    vcf_threat_actions.append(int(threats[t, 0] * 15 + threats[t, 1]))

        if self.cpp_tree.is_root_expanded():
            if dirichlet_alpha > 0:
                actions = self.cpp_tree.get_root_actions()
                noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
                self.cpp_tree.add_dirichlet_noise(0.25, actions, [float(x) for x in noise])
            
            if len(vcf_threat_actions) > 0:
                self.cpp_tree.apply_vcf_bias(vcf_threat_actions, 0.3)

        for _ in range(self.n_playout):
            if self.cpp_tree.is_root_proven_win() or self.cpp_tree.is_root_proven_loss():
                break
            self.playout(env, predict_fn)

        if not self.cpp_tree.is_root_expanded():
            return [], []

        if self.cpp_tree.is_root_proven_win():
            actions = self.cpp_tree.get_root_actions()
            probs = np.zeros(len(actions))
            for i, act in enumerate(actions):
                if self.cpp_tree.is_child_proven_loss(act):
                    probs[i] = 1.0
                    return actions, probs

        action_probs = self.cpp_tree.get_root_action_probs(temperature)
        acts = [p[0] for p in action_probs]
        probs = np.array([p[1] for p in action_probs], dtype=np.float64)
        
        # 嚴格歸一化以滿足 np.random.choice 的要求
        if probs.sum() > 0:
            probs /= probs.sum()
        
        return acts, probs

    def update_with_move(self, last_move):
        """
        ♻️ Subtree Retention (保留子樹)：
        在真實棋盤落下一子後，不砍掉整棵樹，而是將指標移動到下一層，完美繼承算力！
        """
        if last_move != -1:
            self.cpp_tree.update_with_move(last_move)
