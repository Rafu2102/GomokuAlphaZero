"""
train_gomoku.py - Phase 4: AlphaZero 煉丹飛輪 & 競技場
======================================================
完整訓練管線：Self-Play -> Train -> Arena -> Loop
單 GPU 同步管道架構，為 i7-12700 + RTX 4060 量身打造。
包含全自動動態課程排程與專屬終端機即時儀表板 (Dashboard)。
"""

import os
import sys
import time
import gc
# 強制 Windows 終端機使用 UTF-8 輸出，避免 Emoji 導致 cp950 編碼錯誤崩潰
enc = getattr(sys.stdout, 'encoding', None)
if not enc or enc.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 強制在 Windows 啟動 VT100 ANSI 轉義序列，修復儀表板無限洗版問題
if os.name == 'nt':
    os.system('')

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process, Queue

from env import GomokuEnv
from mcts import MCTSEngine, LOCAL_WORKER_CACHE
from resnet import PolicyValueNet
from prediction_server import (
    SharedMemoryPool, PredictionServer, PredictionClient,
    BOARD_SIZE, ACTION_SIZE, INPUT_CHANNELS,
)
from self_play import play_one_game, ReplayBuffer


# ==========================================
# UI Dashboard
# ==========================================
class Dashboard:
    def __init__(self, total_iters):
        self.total_iters = total_iters
        self.start_time = time.time()
        self.iteration = 0
        self.phase_name = "Initializing..."
        self.status = "System Booting..."
        
        self.workers = 0
        self.playouts = 0
        self.batch_size = 0
        self.c_puct = 0.0
        
        self.buffer_size = 0
        self.max_buffer = 0
        
        # 最近一輪勝負
        self.b_wins = 0
        self.w_wins = 0
        self.draws = 0
        self.avg_moves = 0
        self.draw_rate = 0.0
        self.sp_speed = 0.0          # games/min
        
        # 模型表現
        self.total_loss = 0.0
        self.p_loss = 0.0
        self.v_loss = 0.0
        self.aux_loss = 0.0
        self.lr = 0.0
        self.best_win_rate = 0.0
        self.win_rate_threshold = 0.49
        self.grad_norm = 0.0         # 梯度範數
        self.policy_entropy = 0.0    # 策略熵
        
        # Arena
        self.arena_patience = 0      # 連敗計數
        self.arena_last_wr = 0.0     # 上次勝率
        
        # 系統
        self.gpu_mem_used = 0.0      # GPU MB
        self.gpu_mem_total = 0.0     # GPU MB
        self.iter_time = 0.0         # 秒/輪
        
        # 開啟 Windows ANSI 支援 (防閃爍刷屏)
        os.system("")
        
    def format_time(self, seconds):
        if seconds < 0: return "Unknown"
        d = int(seconds // 86400)
        h = int((seconds % 86400) // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if d > 0:
            return f"{d}d {h:02d}:{m:02d}:{s:02d}"
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _gpu_bar(self):
        """GPU 記憶體使用率條"""
        if self.gpu_mem_total <= 0:
            return "N/A"
        pct = self.gpu_mem_used / self.gpu_mem_total * 100
        bar_len = 20
        filled = int((pct / 100) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"[{bar}] {self.gpu_mem_used:.0f}/{self.gpu_mem_total:.0f} MB ({pct:.0f}%)"

    def _buffer_bar(self):
        """Buffer 使用率條"""
        if self.max_buffer <= 0:
            return "N/A"
        pct = self.buffer_size / self.max_buffer * 100
        bar_len = 20
        filled = int((pct / 100) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"[{bar}] {self.buffer_size:,} ({pct:.1f}%)"

    def update(self):
        elapsed = time.time() - self.start_time
        iters_done = max(1, self.iteration)
        time_per_iter = elapsed / iters_done
        eta = time_per_iter * (self.total_iters - self.iteration)
        
        pct = (self.iteration / self.total_iters) * 100
        bar_len = 50
        filled = int((pct / 100) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        iter_speed = f"{self.iter_time:.1f}s" if self.iter_time > 0 else "--"
        sp_speed = f"{self.sp_speed:.1f}" if self.sp_speed > 0 else "--"
        
        box = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   🎮 ALPHAZERO 五子棋 v7「純零」- 階段: {self.phase_name[:33]:<33} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ⚙️ 系統資訊                                                                 ║
║  顯示卡記憶體:  {self._gpu_bar():<60} ║
║  工作程序: {self.workers:<2} | 搜尋量: {self.playouts:<4} | 探索: {self.c_puct:<3.1f} | 批次: {self.batch_size:<4}                  ║
║  每輪耗時:      {iter_speed:<8} | 對弈速度: {sp_speed:<5} 局/分                     ║
╠──────────────────────────────────────────────────────────────────────────────╣
║                                                                              ║
║  📊 訓練狀態                                                                 ║
║  輪次:          {self.iteration:<4} / {self.total_iters:<4}                                                  ║
║  經驗池:        {self._buffer_bar():<58} ║
║  目前狀態:      {self.status[:58]:<58} ║
║                                                                              ║
║  🎯 自我對弈（最近一輪）                                                     ║
║  黑勝: {self.b_wins:<3} | 白勝: {self.w_wins:<3} | 和棋: {self.draws:<3} | 和棋率: {self.draw_rate*100:<5.1f}%                  ║
║  平均手數:      {self.avg_moves:<3} | 對弈速度: {sp_speed:<5} 局/分                     ║
║                                                                              ║
║  🧠 模型表現                                                                 ║
║  損失  總計:    {self.total_loss:<8.4f}  策略: {self.p_loss:<8.4f}                                 ║
║        價值:    {self.v_loss:<8.4f}  輔助: {self.aux_loss:<8.4f}                                 ║
║  梯度範數:      {self.grad_norm:<8.4f}  策略熵: {self.policy_entropy:<8.4f}                             ║
║  學習率:        {self.lr:<.6f}                                                    ║
║                                                                              ║
║  🏆 擂台賽                                                                   ║
║  最佳勝率:      {self.best_win_rate * 100:<5.1f}% | 上次: {self.arena_last_wr * 100:<5.1f}% | 連敗: {self.arena_patience}/3            ║
║  篡位門檻:      {self.win_rate_threshold * 100:.1f}%                                                     ║
║                                                                              ║
║  ⏱️ 時間                                                                     ║
║  已耗時:        {self.format_time(elapsed):<20} 預估剩餘: {self.format_time(eta):<15}   ║
║                                                                              ║
║  📈 整體進度                                                                 ║
║  [{bar}] {pct:>5.1f}%        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        if not hasattr(self, 'last_update_time'):
            self.last_update_time = 0
            
        current_time = time.time()
        # 節流防卡：每 3 秒才真正刷新一次畫面
        if current_time - self.last_update_time < 3.0:
            return
            
        self.last_update_time = current_time
        
        # 直接呼叫系統指令清屏 (最穩定的防刷屏方式，徹底解決一直跳新區塊的問題)
        os.system('cls' if os.name == 'nt' else 'clear')
        sys.stdout.write(box)
        sys.stdout.flush()


# ==========================================
# Config
# ==========================================
class Config:
    # --- Self-Play ---
    num_workers = 4
    games_per_iteration = 24
    n_playout = 400
    c_puct = 5.0
    temp_threshold = 12
    dirichlet_alpha = 0.3

    # --- Training ---
    batch_size = 512
    train_steps_per_iter = 200
    learning_rate = 1e-3  # 更安全的起步步伐，防止初期的 Loss Spike
    lr_min = 1e-6
    weight_decay = 1e-4

    # --- Replay Buffer ---
    buffer_max_size = 1_000_000  # 3000輪長征必備：100萬筆超大記憶池
    min_buffer_for_train = 5_000

    # --- Arena ---
    arena_games = 40
    arena_interval = 5
    arena_n_playout = 400
    win_rate_threshold = 0.49  # 接受平局(0.50)以打破防守型死結循環

    # --- General ---
    max_iterations = 3000  # 3000 輪穩定發育版神級極限長征
    checkpoint_dir = './checkpoints'
    log_dir = './logs/alphazero'


# ==========================================
# Direct GPU Predict (for Arena, no Server)
# ==========================================
def make_predict_fn(model, device):
    model.eval()

    def predict(state_tensor):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                t = torch.from_numpy(state_tensor).unsqueeze(0).to(device)
                log_probs, value = model(t)
            probs = torch.exp(log_probs).cpu().numpy()[0]
            v = float(value.cpu().numpy()[0][0])
        return list(enumerate(probs)), v

    return predict


# ==========================================
# Self-Play Worker
# ==========================================
def _self_play_worker(worker_id, shm_names, condition, pending_slots,
                      pending_count, worker_event, n_games, n_playout,
                      c_puct, temp_threshold, dirichlet_alpha, result_queue):
    # 🛡️ 在子進程中忽略 Ctrl+C (SIGINT)，讓主進程統一處理中斷並叫子進程 terminate
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    client = None
    try:
        # 🔥 Numba JIT 預熱：在 Worker 中首次 import 時會觸發 Numba 編譯
        # 提前跑一個小棋盤操作來強制編譯完畢，避免第一局 timeout
        from env import GomokuEnv as _WarmupEnv
        _w = _WarmupEnv()
        _w.reset()
        _w.step(112)
        _ = _w.get_threat_target()  # 觸發 _generate_threat_heatmap 的 JIT
        _ = _w.get_legal_moves()     # 觸發 _generate_forbidden_mask 的 JIT
        del _w

        client = PredictionClient(
            worker_id=worker_id, shm_names=shm_names, condition=condition,
            pending_slots=pending_slots, pending_count=pending_count, worker_event=worker_event,
        )
        for _ in range(n_games):
            game_data, winner, moves = play_one_game(
                client.predict_for_mcts,
                n_playout=n_playout, c_puct=c_puct,
                temp_threshold=temp_threshold, dirichlet_alpha=dirichlet_alpha,
            )
            result_queue.put((game_data, winner, moves))
            
            # 🚀 修復子進程記憶體暴增：每盤下完立即清空快取並回收 MCTS 孤星節點
            LOCAL_WORKER_CACHE.cache.clear()
            gc.collect()
    except Exception as e:
        import traceback
        crash_msg = f"Worker {worker_id} crashed:\n{traceback.format_exc()}"
        try:
            with open(f"worker_crash_{worker_id}.log", "w", encoding="utf-8") as f:
                f.write(crash_msg)
        except Exception:
            pass
        import sys
        print(crash_msg, file=sys.stderr, flush=True)
        os._exit(1)
    finally:
        if client is not None:
            client.close()


# ==========================================
# Self-Play Pipeline (Multiprocessing)
# ==========================================
def run_self_play(model, config, replay_buffer, dashboard):
    pool = SharedMemoryPool(max_workers=config.num_workers)
    server = PredictionServer(model, pool, max_workers=config.num_workers)
    server.start()
    time.sleep(0.3)

    sync = server.get_sync_primitives()
    shm_names = pool.get_shm_names()

    result_queue = Queue()
    games_per_worker = config.games_per_iteration // config.num_workers

    processes = []
    for i in range(config.num_workers):
        p = Process(
            target=_self_play_worker,
            args=(
                i, shm_names, sync['condition'], sync['pending_slots'],
                sync['pending_count'], sync['worker_events'][i],
                games_per_worker, config.n_playout, config.c_puct,
                config.temp_threshold, config.dirichlet_alpha, result_queue,
            ),
        )
        processes.append(p)
        p.start()

    total_expected = config.num_workers * games_per_worker
    results = []
    
    try:
        # 輪詢接收，同時更新儀表板
        while len(results) < total_expected:
            try:
                res = result_queue.get(timeout=1.0)
                results.append(res)
            except Exception:
                # Worker 異常離開偵測：判斷是否真的 crash (exitcode != 0)，而不是正常提早結束
                for p in processes:
                    if not p.is_alive() and p.exitcode is not None and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker {p.pid} crashed with exitcode {p.exitcode}! 已完成 {len(results)}/{total_expected} 盤。"
                        )
                
            if dashboard:
                dashboard.status = f"[Phase A] 收集資料: 對局 {len(results)}/{total_expected} 完成..."
                dashboard.update()

        for p in processes:
            p.join(timeout=30)
    finally:
        # 🛡️ 絕對防線：不管是正常結束還是 Ctrl+C 中斷，
        # 都必須強制終結所有子進程並釋放共享記憶體。
        for p in processes:
            if p.is_alive():
                p.terminate()
        server.stop()
        pool.cleanup()

    stats = {'black_wins': 0, 'white_wins': 0, 'draws': 0, 'total_moves': 0}
    for game_data, winner, moves in results:
        replay_buffer.add_game(game_data)
        stats['total_moves'] += moves
        if winner == 1:
            stats['black_wins'] += 1
        elif winner == -1:
            stats['white_wins'] += 1
        else:
            stats['draws'] += 1

    return stats


# ==========================================
# AlphaZero Loss (Train Step)
# ==========================================
def train_step(model, optimizer, scaler, states, target_probs, target_values,
               target_threats, device):
    """AlphaZero v7 訓練步驟：Policy + Value + Aux Threat Loss"""
    model.train()
    s = torch.FloatTensor(states).to(device)
    pi = torch.FloatTensor(target_probs).to(device)
    z = torch.FloatTensor(target_values).to(device)
    t = torch.FloatTensor(target_threats).to(device)  # [B, 15, 15] Aux Target

    optimizer.zero_grad()
    
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        log_p, v, aux = model(s, return_aux=True)  # 啟用 Aux Head
        # AlphaZero 標準交叉熵
        policy_loss = -torch.sum(pi * log_p) / pi.size(0)
        value_loss = F.mse_loss(v.view(-1), z.view(-1))
        # Aux Loss：讓 backbone 學會辨識棋型，權重 0.15 (溫和引導，不壓過主任務)
        aux_loss = F.mse_loss(aux, t)
        loss = policy_loss + 1.0 * value_loss + 0.15 * aux_loss

    scaler.scale(loss).backward()
    
    # 📀 梯度範數監控 (Gradient Norm) — 在 unscale 後計算
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    scaler.step(optimizer)
    scaler.update()

    # 📀 策略熵 (Policy Entropy) — 判斷策略是否崩潰
    with torch.no_grad():
        probs = torch.exp(log_p)  # log_softmax -> softmax
        entropy = -(probs * log_p).sum(dim=1).mean()  # H(p) = -Σ p*log(p)

    return {
        'total_loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'aux_loss': aux_loss.item(),
        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        'policy_entropy': entropy.item(),
    }


# ==========================================
# Arena (Candidate vs Best)
# ==========================================
def play_arena_game(black_predict, white_predict, n_playout=200, c_puct=5.0):
    env = GomokuEnv()
    env.reset()
    black_mcts = MCTSEngine(c_puct=c_puct, n_playout=n_playout)
    white_mcts = MCTSEngine(c_puct=c_puct, n_playout=n_playout)

    while not env.done:
        if env.current_player == 1:
            mcts, predict_fn = black_mcts, black_predict
        else:
            mcts, predict_fn = white_mcts, white_predict

        acts, probs = mcts.get_action_probs(
            env, predict_fn, temperature=1e-3, dirichlet_alpha=0,
        )
        
        # 🛡️ 防禦：MCTS 回傳空結果 (終局邊界)
        if len(acts) == 0:
            break
        
        action = acts[np.argmax(probs)]

        env.step(action)
        black_mcts.update_with_move(action)
        white_mcts.update_with_move(action)

    return env.winner


def arena_evaluate(candidate_model, best_model, config, device, dashboard=None):
    candidate_predict = make_predict_fn(candidate_model, device)
    best_predict = make_predict_fn(best_model, device)

    half = config.arena_games // 2
    candidate_wins = 0
    candidate_draws = 0

    for i in range(half):
        if dashboard:
            dashboard.status = f"[Phase C] Arena (Cand as Black): 挑戰 {i+1}/{half} 局..."
            dashboard.update()
        winner = play_arena_game(
            candidate_predict, best_predict,
            n_playout=config.arena_n_playout, c_puct=config.c_puct,
        )
        if winner == 1: candidate_wins += 1
        elif winner == 0: candidate_draws += 1

    for i in range(half):
        if dashboard:
            dashboard.status = f"[Phase C] Arena (Cand as White): 挑戰 {i+1}/{half} 局..."
            dashboard.update()
        winner = play_arena_game(
            best_predict, candidate_predict,
            n_playout=config.arena_n_playout, c_puct=config.c_puct,
        )
        if winner == -1: candidate_wins += 1
        elif winner == 0: candidate_draws += 1

    total = config.arena_games
    win_rate = (candidate_wins + 0.5 * candidate_draws) / total
    return {
        'win_rate': win_rate,
        'wins': candidate_wins,
        'draws': candidate_draws,
        'losses': total - candidate_wins - candidate_draws,
    }


# ==========================================
# Checkpoint
# ==========================================
def save_checkpoint(path, model, optimizer, scheduler, iteration):
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # strict=False 允許跳過 aux_head 的缺失權重（相容舊 checkpoint）
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('iteration', 0)


# ==========================================
# Main Pipeline
# ==========================================
def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 1. Dashboard
    dash = Dashboard(config.max_iterations)
    
    # 2. Model & AMP
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    latest_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')

    model = PolicyValueNet().to(device)
    best_model = PolicyValueNet().to(device)
    scaler = torch.amp.GradScaler(device='cuda')

    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.max_iterations, eta_min=config.lr_min,
    )

    start_iteration = 0
    # 優先讀取 latest_model 恢復進度 (這包含了被 Ctrl+C 中斷的進度)
    if os.path.exists(latest_path):
        start_iteration = load_checkpoint(
            latest_path, model, optimizer, scheduler, device
        )
        dash.status = f"System Boot: Resumed from latest iteration {start_iteration}"
        # 嘗試讀取 best_model 到對抗基準
        if os.path.exists(best_model_path):
            best_model_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            best_model.load_state_dict(best_model_checkpoint['model_state_dict'])
        else:
            best_model.load_state_dict(model.state_dict())
    elif os.path.exists(best_model_path):
        start_iteration = load_checkpoint(
            best_model_path, model, optimizer, scheduler, device
        )
        best_model.load_state_dict(model.state_dict())
        dash.status = f"System Boot: Resumed from best iteration {start_iteration}"
    else:
        best_model.load_state_dict(model.state_dict())
        dash.status = "System Boot: Starting from scratch"

    # 3. Buffer & TensorBoard
    replay_buffer = ReplayBuffer(max_size=config.buffer_max_size)
    writer = SummaryWriter(config.log_dir)
    global_train_step = start_iteration * config.train_steps_per_iter

    # 4. Main Loop
    patience_counter = 0
    is_ice_breaking = False
    try:
        for iteration in range(start_iteration, config.max_iterations):
            iter_start = time.time()
            dash.iteration = iteration + 1
            
            # --- 曲率驅動動態更新 (3000 輪・i7-12700H 14C/20T + 32GB + RTX 4060 8GB 極限壓榨版) ---
            if config.max_iterations > 10:
                if iteration < 200:
                    # Phase A: 極速拓荒，瘋狂累積資料量
                    config.num_workers = 8
                    config.games_per_iteration = 40
                    config.n_playout = 200
                    config.train_steps_per_iter = 100
                    config.batch_size = 512
                    config.arena_interval = 10
                    config.c_puct = 5.0
                    config.temp_threshold = 15
                    dash.phase_name = "Phase A: 快速拓荒 (8核海量對局・廣泛建構開局庫)"
                elif iteration < 800:
                    # Phase B: 平穩學習，逐漸加深戰術防守
                    config.num_workers = 8
                    config.games_per_iteration = 32
                    config.n_playout = 400
                    config.train_steps_per_iter = 200
                    config.batch_size = 512
                    config.arena_interval = 5
                    config.c_puct = 4.0
                    config.temp_threshold = 10
                    dash.phase_name = "Phase B: 中盤拉扯 (8核深化・防守反擊與活三訓練)"
                elif iteration < 1600:
                    # Phase C: 精細戰略，降低 Worker 防止進程碎片化
                    config.num_workers = 6
                    config.games_per_iteration = 28
                    config.n_playout = 800
                    config.train_steps_per_iter = 250  # 🛡️ 防 Overfitting（原 300 過高）
                    config.batch_size = 512
                    config.arena_interval = 3
                    config.c_puct = 3.0
                    config.temp_threshold = 6
                    dash.phase_name = "Phase C: 算力解放 (6核精算・深入推演 VCF 連殺)"
                else:
                    # Phase D: 絕對收斂期，Playout 1000 即可突破天花板
                    config.num_workers = 6
                    config.games_per_iteration = 24
                    config.n_playout = 1000
                    config.train_steps_per_iter = 250  # 🛡️ 後期微調即可，避免權重崩潰
                    config.batch_size = 512
                    config.arena_interval = 3
                    config.c_puct = 3.0                # 🛡️ 保持防守盲區探索力
                    config.temp_threshold = 5
                    dash.phase_name = "Phase D: 死神領域 (6核深算・防禦力極限打磨)"
            else:
                dash.phase_name = "Smoke Test Mode"

            # 破冰機制覆蓋 (如果正在破冰，強制提高探索)
            if is_ice_breaking:
                config.dirichlet_alpha = 1.0
                config.temp_threshold = 12
            else:
                config.dirichlet_alpha = 0.3

            # 同步面板狀態
            dash.workers = config.num_workers
            dash.playouts = config.n_playout
            dash.batch_size = config.batch_size
            dash.c_puct = config.c_puct
            dash.max_buffer = config.buffer_max_size
            dash.buffer_size = len(replay_buffer)
            dash.lr = optimizer.param_groups[0]['lr']
            
            # ⚡ GPU 記憶體監控
            if torch.cuda.is_available():
                dash.gpu_mem_used = torch.cuda.memory_allocated(device) / 1024**2
                dash.gpu_mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**2
            
            iter_start_time = time.time()  # ⭐ 計時器
            dash.update()

            # -- Phase A: Self-Play --
            model.eval()
            sp_start = time.time()
            sp_stats = run_self_play(model, config, replay_buffer, dash)
            sp_elapsed = time.time() - sp_start

            total_games = sp_stats['black_wins'] + sp_stats['white_wins'] + sp_stats['draws']
            dash.avg_moves = sp_stats['total_moves'] // max(1, total_games)
            dash.b_wins = sp_stats['black_wins']
            dash.w_wins = sp_stats['white_wins']
            dash.draws = sp_stats['draws']
            dash.buffer_size = len(replay_buffer)
            dash.draw_rate = dash.draws / max(1, total_games)
            dash.sp_speed = total_games / max(0.01, sp_elapsed / 60.0)  # games/min

            # 👇 --- 自動反平局防禦機制 (附自動冷卻) --- 👇
            draw_rate = dash.draw_rate
            if draw_rate > 0.3 or dash.avg_moves > 120:
                is_ice_breaking = True
                dash.status = f"[警報] 陷入和平主義 (平局率 {draw_rate*100:.1f}%)！下一輪強制觸發破冰！"
                dash.update()
                time.sleep(2)
            else:
                if patience_counter < 3:
                    is_ice_breaking = False
            # 👆 ----------------------------------------- 👆

            # 📀 Self-Play TensorBoard
            writer.add_scalar('SelfPlay/Black_Wins', sp_stats['black_wins'], iteration)
            writer.add_scalar('SelfPlay/White_Wins', sp_stats['white_wins'], iteration)
            writer.add_scalar('SelfPlay/Draws', sp_stats['draws'], iteration)
            writer.add_scalar('SelfPlay/Draw_Rate', dash.draw_rate, iteration)
            writer.add_scalar('SelfPlay/Avg_Moves', dash.avg_moves, iteration)
            writer.add_scalar('SelfPlay/Buffer_Size', len(replay_buffer), iteration)
            writer.add_scalar('SelfPlay/Buffer_Util', len(replay_buffer) / config.buffer_max_size, iteration)
            writer.add_scalar('SelfPlay/Games_Per_Min', dash.sp_speed, iteration)
            writer.add_scalar('SelfPlay/Total_Games', total_games, iteration)
            
            LOCAL_WORKER_CACHE.cache.clear()

            # -- Phase B: Training --
            if len(replay_buffer) >= config.min_buffer_for_train:
                epoch_losses = {'total': [], 'policy': [], 'value': [], 'aux': [],
                               'grad_norm': [], 'entropy': []}
                for step in range(config.train_steps_per_iter):
                    
                    if step % max(1, config.train_steps_per_iter // 20) == 0:
                        dash.status = f"[Phase B] Training: 梯度下降 {step+1}/{config.train_steps_per_iter}..."
                        dash.update()

                    states, probs, values, threats = replay_buffer.sample_batch(config.batch_size)
                    losses = train_step(model, optimizer, scaler, states, probs, values, threats, device)
                    
                    epoch_losses['total'].append(losses['total_loss'])
                    epoch_losses['policy'].append(losses['policy_loss'])
                    epoch_losses['value'].append(losses['value_loss'])
                    epoch_losses['aux'].append(losses['aux_loss'])
                    epoch_losses['grad_norm'].append(losses['grad_norm'])
                    epoch_losses['entropy'].append(losses['policy_entropy'])

                    global_train_step += 1
                    # 📀 每步細粒度 Loss (per-step)
                    writer.add_scalar('Loss/Total', losses['total_loss'], global_train_step)
                    writer.add_scalar('Loss/Policy', losses['policy_loss'], global_train_step)
                    writer.add_scalar('Loss/Value', losses['value_loss'], global_train_step)
                    writer.add_scalar('Loss/Aux', losses['aux_loss'], global_train_step)
                    writer.add_scalar('Train/GradNorm', losses['grad_norm'], global_train_step)
                    writer.add_scalar('Train/PolicyEntropy', losses['policy_entropy'], global_train_step)

                scheduler.step()
                dash.lr = optimizer.param_groups[0]['lr']
                dash.total_loss = np.mean(epoch_losses['total'])
                dash.p_loss = np.mean(epoch_losses['policy'])
                dash.v_loss = np.mean(epoch_losses['value'])
                dash.aux_loss = np.mean(epoch_losses['aux'])
                dash.grad_norm = np.mean(epoch_losses['grad_norm'])
                dash.policy_entropy = np.mean(epoch_losses['entropy'])

                # 📀 每輪平均 Loss (per-iteration, 更平滑)
                writer.add_scalar('Epoch/Total_Loss', dash.total_loss, iteration)
                writer.add_scalar('Epoch/Policy_Loss', dash.p_loss, iteration)
                writer.add_scalar('Epoch/Value_Loss', dash.v_loss, iteration)
                writer.add_scalar('Epoch/Aux_Loss', dash.aux_loss, iteration)
                writer.add_scalar('Epoch/GradNorm', dash.grad_norm, iteration)
                writer.add_scalar('Epoch/PolicyEntropy', dash.policy_entropy, iteration)
                writer.add_scalar('LR', dash.lr, iteration)
                
                # 訓練完即可存為最新版
                save_checkpoint(latest_path, model, optimizer, scheduler, iteration + 1)
            else:
                dash.status = f"[Phase B] Training SKIPPED (buffer {len(replay_buffer)} < {config.min_buffer_for_train})"
                dash.update()

            # -- Phase C: Arena --
            if (iteration + 1) % config.arena_interval == 0:
                arena_result = arena_evaluate(model, best_model, config, device, dash)
                wr = arena_result['win_rate']
                dash.arena_last_wr = wr
                
                writer.add_scalar('Arena/Win_Rate', wr, iteration)
                writer.add_scalar('Arena/Patience', patience_counter, iteration)

                if wr >= config.win_rate_threshold:
                    patience_counter = 0
                    is_ice_breaking = False
                    config.dirichlet_alpha = 0.3
                    
                    dash.best_win_rate = wr
                    dash.status = f"[{dash.phase_name[:7]}] >>> 新王誕生! 勝率 {wr*100:.1f}% 篡位成功! <<<"
                    best_model.load_state_dict(model.state_dict())
                    save_checkpoint(best_model_path, best_model, optimizer, scheduler, iteration + 1)
                    torch.save({'model_state_dict': best_model.state_dict()}, 
                               os.path.join(config.checkpoint_dir, f'model_v{iteration + 1}.pth'))
                else:
                    patience_counter += 1
                    dash.status = f"[{dash.phase_name[:7]}] >>> 踢館失敗 ({patience_counter}連敗). 勝率 {wr*100:.1f}%."
                    # 回滾模型，並同步重建 optimizer/scheduler 避免殘留動量拉扯
                    model.load_state_dict(best_model.state_dict())
                    optimizer = AdamW(
                        model.parameters(), lr=config.learning_rate,
                        weight_decay=config.weight_decay,
                    )
                    scheduler = CosineAnnealingLR(
                        optimizer, T_max=config.max_iterations - iteration,
                        eta_min=config.lr_min,
                    )
                    
                    if patience_counter >= 3:
                        is_ice_breaking = True
                        dash.status = f"[{dash.phase_name[:7]}] 踢館 >=3 連敗! 啟動破冰(提升噪聲)!"
                        config.dirichlet_alpha = 1.0
                dash.arena_patience = patience_counter
                writer.add_scalar('Arena/Best_Win_Rate', dash.best_win_rate, iteration)
                dash.update()
                time.sleep(1.5)
                
            # 👇 --- 新增：記憶體強制清道夫 --- 👇
            LOCAL_WORKER_CACHE.cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 更新 GPU 記憶體狀態
                dash.gpu_mem_used = torch.cuda.memory_allocated(device) / 1024**2
            # 👆 ---------------------------- 👆

            # ⭐ 記錄此輪耗時
            dash.iter_time = time.time() - iter_start_time
            writer.add_scalar('System/Iter_Time_Sec', dash.iter_time, iteration)
            if torch.cuda.is_available():
                writer.add_scalar('System/GPU_Memory_MB', dash.gpu_mem_used, iteration)
            writer.add_scalar('System/Buffer_Size', len(replay_buffer), iteration)

    except KeyboardInterrupt:
        dash.status = "[!] USER INTERRUPT: 安全存檔中..."
        dash.update()
        # 🛡️ 防護：如果中斷發生在第一輪 self-play 期間，iteration 尚未被賦值
        save_iter = iteration + 1 if 'iteration' in dir() else start_iteration
        save_checkpoint(latest_path, model, optimizer, scheduler, save_iter)
        
        # 強制退出整個 Python 進程樹 (不等待垃圾回收，光速閃退)
        os._exit(0)
    finally:
        writer.close()
        dash.status = "[√] 訓練已安全結束！"
        dash.update()


# ==========================================
# Smoke Test / Launch
# ==========================================
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    if '--test' in sys.argv:
        Config.num_workers = 2
        Config.games_per_iteration = 2
        Config.n_playout = 8
        Config.train_steps_per_iter = 5
        Config.batch_size = 32
        Config.min_buffer_for_train = 10
        Config.arena_interval = 1
        Config.arena_games = 4
        Config.arena_n_playout = 8
        Config.max_iterations = 1
        Config.checkpoint_dir = './checkpoints_test'
        Config.log_dir = './logs/smoke_test'

    sys.exit(main())
