"""
mcts.py - v7 Pure Zero MCTS 搜尋引擎
========================================
包含：MCTS 節點、Zobrist 轉置表、VCF Root-only Candidate Filter、
      Additive Prior Bias、Proof Propagation、Opening Book 支援。
"""
import numpy as np
import math
import random

# ==========================================
# 🔮 Zobrist 轉置表全域快取 (Transposition Table)
# ==========================================
class ZobristCache:
    """存放神經網路推論結果，遇到「殊途同歸」的盤面直接提取，節省 GPU 推論"""
    def __init__(self):
        self.cache = {}
    
    def get(self, state_bytes):
        return self.cache.get(state_bytes, None)
        
    def save(self, state_bytes, policy, value):
        self.cache[state_bytes] = (policy, value)

# 在多進程 Async Self-Play 中，每個 Worker 擁有獨立的快取副本。
LOCAL_WORKER_CACHE = ZobristCache()

# ==========================================
# 🌳 MCTS 節點與終局證明器 (Proof Propagation)
# ==========================================
class MCTSNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {} # action : MCTSNode
        self.n_visits = 0  # 訪問次數 (N)
        self.q_value = 0.0 # 平均勝率 (Q)
        self.u_value = 0.0 # 探索價值 (U)
        self.p_prior = prior_p # 神經網路先驗機率 (P)
        self.original_p = prior_p # 不可變原始副本 (防止 Dirichlet 污染)
        
        # ♾️ MCTS Solver: 絕對證明標籤
        self.is_proven_win = False
        self.is_proven_loss = False

    def expand(self, action_probs):
        """將神經網路產生的合法步機率轉換為樹的子節點"""
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob)

    def is_expanded(self):
        return len(self.children) > 0

    def get_value(self, c_puct):
        """計算 UCB 值：Q + U，支援 Proof Propagation"""
        self.u_value = (c_puct * self.p_prior * 
                        math.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        
        # 視角關鍵：Parent 在評估 Child
        # Child.is_proven_loss = 進入該狀態的玩家(對手)必敗 → 對 Parent 是殺招
        # Child.is_proven_win  = 進入該狀態的玩家(對手)必勝 → 對 Parent 是死路
        if self.is_proven_loss:
             return float('inf')
        if self.is_proven_win:
             return float('-inf')

        return self.q_value + self.u_value

    def update(self, leaf_value):
        """反向傳播更新 N 與 Q"""
        self.n_visits += 1
        self.q_value += 1.0 * (leaf_value - self.q_value) / self.n_visits

# ==========================================
# 🧠 動態 MCTS 搜尋引擎 (Dynamic MCTS Engine)
# ==========================================
class MCTSEngine:
    def __init__(self, c_puct=5.0, n_playout=800, opening_book=None):
        self.root = MCTSNode(None, 1.0)
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.opening_book = opening_book  # 開局庫實例 (None = 不使用)

    def playout(self, env, predict_fn):
        """執行【單次】腦內模擬：Select -> Expand -> Backup。不碰 VCF。"""
        node = self.root
        
        # 1. Selection (選擇)
        while node.is_expanded():
            action, node = max(node.children.items(), 
                             key=lambda item: item[1].get_value(self.c_puct))
            env.step_fast(action)

        # 2. Expansion & Evaluation (擴展與評估)
        state_tensor = env._get_obs_fast()  # 統一 4ch 觀測
        cache_key = (int(env.zobrist_hash), env.current_player, env.last_move)
        
        is_game_ended = env.done
        leaf_value = 0.0
        
        if not is_game_ended:
            cached_result = LOCAL_WORKER_CACHE.get(cache_key)
            if cached_result is not None:
                action_probs, leaf_value = cached_result
            else:
                action_probs, leaf_value = predict_fn(state_tensor)
                
                # 過濾不合法步，並重新歸一化 (Masking)
                legal_moves = env.get_legal_moves()
                action_probs = [(a, p) for a, p in action_probs if a in legal_moves]
                total_p = sum([p for a, p in action_probs])
                if total_p > 0:
                    action_probs = [(a, p / total_p) for a, p in action_probs]
                elif len(action_probs) > 0:
                    prob = 1.0 / len(action_probs)
                    action_probs = [(a, prob) for a, _ in action_probs]
                
                LOCAL_WORKER_CACHE.save(cache_key, action_probs, leaf_value)
                
            node.expand(action_probs)
        else:
            # MCTS Solver (終局證明)
            if env.winner == env.current_player:
                leaf_value = -1.0
                node.is_proven_loss = True
            elif env.winner == -env.current_player:
                leaf_value = 1.0
                node.is_proven_win = True
            else:
                leaf_value = 0.0

        # 3. Backup (回溯更新)
        current_node = node
        current_val = -leaf_value
        while current_node is not None:
            current_node.update(current_val)
            
            # ♾️ MCTS Solver Proof Propagation
            if current_node.is_expanded():
                if any(child.is_proven_loss for child in current_node.children.values()):
                    current_node.is_proven_win = True
                    current_node.is_proven_loss = False
                elif all(child.is_proven_win for child in current_node.children.values()):
                    current_node.is_proven_loss = True
                    current_node.is_proven_win = False

            current_node = current_node.parent
            current_val = -current_val

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

        # 🗡️ VCF Root-only Candidate Detection (board.copy 安全)
        vcf_threat_actions = set()
        stone_count = np.count_nonzero(env.board)
        if stone_count >= 20:
            from vcf_vct import vcf_search, _find_four_threats
            vcf_win = vcf_search(env.board.copy(), env.current_player, max_depth=15)
            if vcf_win:
                # ❌ 絕對不能在這裡設 self.root.is_proven_win = True
                #    VCF 是 Soft Bias，必須讓 MCTS playout 去驗證具體殺招
                threats = _find_four_threats(env.board.copy(), env.current_player)
                for t in range(len(threats)):
                    vcf_threat_actions.add(threats[t, 0] * 15 + threats[t, 1])

        # Dirichlet Noise + VCF Additive Bias (三步獨立處理)
        if len(self.root.children) > 0:
            actions = list(self.root.children.keys())

            # 1️⃣ 基礎先驗與噪聲處理
            if dirichlet_alpha > 0:
                noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
                for i, action in enumerate(actions):
                    child = self.root.children[action]
                    child.p_prior = 0.75 * child.original_p + 0.25 * noise[i]
            else:
                # Arena 絕對零度模式：還原純粹先驗，不加任何噪聲
                for action in actions:
                    self.root.children[action].p_prior = self.root.children[action].original_p

            # 2️⃣ 獨立注入 VCF Additive Residual Bias (保證 Arena 也能觸發)
            if len(vcf_threat_actions) > 0:
                vcf_alpha = 0.3
                for action in actions:
                    if action in vcf_threat_actions:
                        self.root.children[action].p_prior += vcf_alpha

            # 3️⃣ 嚴格歸一化 → PUCT 數學正確 (Σp = 1)
            total_prior = sum(c.p_prior for c in self.root.children.values())
            if total_prior > 0:
                for c in self.root.children.values():
                    c.p_prior /= total_prior

        for _ in range(self.n_playout):
            if self.root.is_proven_win or self.root.is_proven_loss:
                break
            env_copy = env.clone()
            self.playout(env_copy, predict_fn)

        # 空節點防禦
        if not self.root.children:
            return [], []

        action_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*action_visits)

        # 🚨 MCTS Solver 必勝覆寫
        if self.root.is_proven_win:
            probs = np.zeros(len(acts))
            for i, act in enumerate(acts):
                if self.root.children[act].is_proven_loss:
                    probs[i] = 1.0
                    return acts, probs
        
        # Temperature (溫度退火)
        if temperature < 1e-2:
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            visit_arr = np.array(visits, dtype=np.float64)
            probs = visit_arr ** (1.0 / temperature)
            probs /= probs.sum()
            
        return acts, probs

    def update_with_move(self, last_move):
        """
        ♻️ Subtree Retention (保留子樹)：
        在真實棋盤落下一子後，不砍掉整棵樹，而是將指標移動到下一層，完美繼承算力！
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTSNode(None, 1.0)
