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
import shutil
# 強制 Windows 終端機使用 UTF-8 輸出，避免 Emoji 導致 cp950 編碼錯誤崩潰
enc = getattr(sys.stdout, 'encoding', None)
if not enc or enc.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 強制在 Windows 啟動 VT100 ANSI 轉義序列，修復儀表板無限洗版問題
if os.name == 'nt':
    os.system('')

# ==========================================
# 📝 雙重日誌記錄器 (確保當機時日誌必定留存於硬碟)
# ==========================================
class DualLogger:
    def __init__(self, filepath, stream):
        self.terminal = stream
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def exception_hook(exctype, value, tb):
    import traceback
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    sys.__stderr__.write(err_msg)
    try:
        with open("train_system.log", "a", encoding="utf-8") as f:
            f.write("\n=== CRASH TRACEBACK ===\n")
            f.write(err_msg)
            f.write("========================\n")
    except Exception:
        pass

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
from self_play import play_one_game, ReplayBuffer, _self_play_worker


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
        
    def _visual_len(self, s):
        """計算字串在終端機下的視覺寬度 (中文/Emoji為2，英文/數字為1)"""
        w = 0
        for char in s:
            if ord(char) > 127:  # 非 ASCII 字元，通常是中文或 Emoji
                w += 2
            else:
                w += 1
        return w

    def _format_line(self, content, total_w=76):
        """自動在 content 後面補足空格，使視覺總寬度恰好為 total_w"""
        cur_w = self._visual_len(content)
        pad = total_w - cur_w
        if pad > 0:
            return content + " " * pad
        return content

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

        # 🛡️ 實時演算法防禦護盾狀態渲染
        warmup_status = f"拓荒暖身中 ({self.buffer_size:,}/50,000)" if self.buffer_size < 50000 else "常規訓練 (已解鎖)"
        if self.buffer_size < 50000:
            step_status = "掛起 (拓荒中)"
        elif self.buffer_size < self.max_buffer * 0.5:
            step_status = "50 步 (溫和暖身)"
        elif self.buffer_size < self.max_buffer * 0.75:
            step_status = "100 步 (平滑發育)"
        elif self.buffer_size < self.max_buffer * 0.9:
            step_status = "160 步 (中盤深化)"
        else:
            if self.iteration < 200:
                steps = 100
            elif self.iteration < 800:
                steps = 120
            else:
                steps = 250
            step_status = f"{steps} 步 (極限修煉)"
        
        def f_line(content):
            return f"║{self._format_line(content, 76)}║"

        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
        lines.append(f_line(f"   🎮 ALPHAZERO 五子棋 v7「純零」- 階段: {self.phase_name}"))
        lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
        lines.append(f_line("  ⚙️ 系統資訊"))
        lines.append(f_line(f"  顯示卡記憶體:  {self._gpu_bar()}"))
        lines.append(f_line(f"  工作程序: {self.workers:<2} | 搜尋量: {self.playouts:<4} | 探索: {self.c_puct:<3.1f} | 批次: {self.batch_size:<4}"))
        lines.append(f_line(f"  每輪耗時:      {iter_speed:<8} | 對弈速度: {sp_speed:<5} 局/分"))
        lines.append("╠──────────────────────────────────────────────────────────────────────────────╣")
        lines.append(f_line("  🛡️ 演算法防禦系統"))
        lines.append(f_line(f"  🔒 溫和拓荒鎖: {warmup_status} | ⚡ 漸進步數: {step_status}"))
        lines.append(f_line("  🛡️ 否決安全鎖: 監控中 (Veto: p_loss > 1.2) | 💾 數據飛輪: replay_buffer.pkl (啟用)"))
        lines.append("╠──────────────────────────────────────────────────────────────────────────────╣")
        lines.append(f_line(""))
        lines.append(f_line("  📊 訓練狀態"))
        lines.append(f_line(f"  輪次:          {self.iteration:<4} / {self.total_iters:<4}"))
        lines.append(f_line(f"  經驗池:        {self._buffer_bar()}"))
        lines.append(f_line(f"  目前狀態:      {self.status}"))
        lines.append(f_line(""))
        lines.append(f_line("  🎯 自我對弈（最近一輪）"))
        lines.append(f_line(f"  黑勝: {self.b_wins:<3} | 白勝: {self.w_wins:<3} | 和棋: {self.draws:<3} | 和棋率: {self.draw_rate*100:<5.1f}%"))
        lines.append(f_line(f"  平均手數:      {self.avg_moves:<3} | 對弈速度: {sp_speed:<5} 局/分"))
        lines.append(f_line(""))
        lines.append(f_line("  🧠 模型表現"))
        lines.append(f_line(f"  損失  總計:    {self.total_loss:<8.4f}  策略: {self.p_loss:<8.4f}"))
        lines.append(f_line(f"        價值:    {self.v_loss:<8.4f}  輔助: {self.aux_loss:<8.4f}"))
        lines.append(f_line(f"  梯度範數:      {self.grad_norm:<8.4f}  策略熵: {self.policy_entropy:<8.4f}"))
        lines.append(f_line(f"  學習率:        {self.lr:<.6f}"))
        lines.append(f_line(""))
        lines.append(f_line("  🏆 擂台賽"))
        lines.append(f_line(f"  最佳勝率:      {self.best_win_rate * 100:<5.1f}% | 上次: {self.arena_last_wr * 100:<5.1f}% | 連敗: {self.arena_patience}/3"))
        lines.append(f_line(f"  篡位門檻:      {self.win_rate_threshold * 100:.1f}%"))
        lines.append(f_line(""))
        lines.append(f_line("  ⏱️ 時間"))
        lines.append(f_line(f"  已耗時:        {self.format_time(elapsed):<20} 預估剩餘: {self.format_time(eta):<15}"))
        lines.append(f_line(""))
        lines.append(f_line("  📈 整體進度"))
        lines.append(f_line(f"  [{bar}] {pct:>5.1f}%"))
        lines.append(f_line(""))
        lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
        box = "\n".join(lines) + "\n"
        if not hasattr(self, 'last_update_time'):
            self.last_update_time = 0
            
        current_time = time.time()
        # 節流防卡：每 3 秒才真正刷新一次畫面
        if current_time - self.last_update_time < 3.0:
            return
            
        self.last_update_time = current_time
        
        # 直接呼叫系統指令清屏
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
    dirichlet_alpha = 0.08

    # --- Training ---
    batch_size = 1024
    train_steps_per_iter = 200
    learning_rate = 2e-4  # 更安全的起步步伐，防止中盤發散
    lr_min = 1e-6
    weight_decay = 1e-4

    # --- Replay Buffer ---
    buffer_max_size = 100_000  # 最佳化：縮減至 100,000 筆，大幅提升數據更新汰換率
    min_buffer_for_train = 5_000

    # --- Arena ---
    arena_games = 40
    arena_interval = 5
    arena_n_playout = 400
    win_rate_threshold = 0.49  # 容許平局(0.50)篡位以打破防守型死結，但結合破冰保護鎖可完全防止隨機退化

    # --- General ---
    max_iterations = 3000  # 3000 輪穩定發育版神級極限長征
    checkpoint_dir = './checkpoints'
    log_dir = './logs/alphazero'

    def __init__(self):
        # 將所有的類別屬性複製為實例屬性，以便在 spawn 多進程序列化時能夠正確傳遞修改後的值
        for key in list(self.__class__.__dict__.keys()):
            if not key.startswith('__'):
                val = getattr(self.__class__, key)
                if not callable(val) and not isinstance(val, property):
                    setattr(self, key, val)


# ==========================================
# Direct GPU Predict (for Arena, no Server)
# ==========================================
def make_predict_fn(model, device):
    model.eval()

    def predict(state_tensor):
        with torch.no_grad():
            if device.type == 'cpu':
                t = torch.from_numpy(state_tensor).unsqueeze(0).to(device)
                log_probs, value = model(t)
            else:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    t = torch.from_numpy(state_tensor).unsqueeze(0).to(device)
                    log_probs, value = model(t)
            probs = torch.exp(log_probs).cpu().numpy()[0]
            v = float(value.cpu().numpy()[0][0])
        return list(enumerate(probs)), v

    return predict



# ==========================================
# Self-Play Pipeline (Multiprocessing)
# ==========================================
def run_self_play(model, config, replay_buffer, dashboard):
    # 🛡️ 確保傳遞給推論伺服器子進程的模型在 CPU 上，防止 CUDA Runtime 跨進程序列化崩潰
    model.cpu()
    print("[run_self_play] Model moved to CPU.", flush=True)
    pool = SharedMemoryPool(max_workers=config.num_workers)
    print(f"[run_self_play] SharedMemoryPool initialized. Name inputs: {pool.in_name}", flush=True)
    server = PredictionServer(model, pool, max_workers=config.num_workers)
    print("[run_self_play] Starting PredictionServer...", flush=True)
    server.start()
    time.sleep(0.3)
    print(f"[run_self_play] PredictionServer started. PID={server.pid}", flush=True)

    sync = server.get_sync_primitives()
    shm_names = pool.get_shm_names()

    result_queue = Queue()
    games_per_worker = config.games_per_iteration // config.num_workers

    processes = []
    print(f"[run_self_play] Spawning {config.num_workers} workers. Games per worker: {games_per_worker}", flush=True)
    for i in range(config.num_workers):
        p = Process(
            target=_self_play_worker,
            args=(
                i, shm_names, sync['request_queue'], sync['worker_events'][i],
                config.num_workers, games_per_worker, config.n_playout, config.c_puct,
                config.temp_threshold, config.dirichlet_alpha, result_queue,
            ),
        )
        processes.append(p)
        p.start()
        print(f"[run_self_play] Worker {i} started. PID={p.pid}", flush=True)
    print("[run_self_play] All workers spawned.", flush=True)

    total_expected = config.num_workers * games_per_worker
    results = []
    last_result_time = time.time()  # 🛡️ 停滯偵測計時器
    STALL_TIMEOUT = 300  # Phase C/D 單局可達 3 分鐘，5 分鐘無結果才視為死鎖
    
    try:
        # 輪詢接收，同時更新儀表板
        while len(results) < total_expected:
            try:
                res = result_queue.get(timeout=1.0)
                results.append(res)
                last_result_time = time.time()  # 收到新結果，重置計時器
            except Exception:
                # Worker 異常離開偵測
                alive_count = sum(1 for p in processes if p.is_alive())
                for p in processes:
                    if not p.is_alive() and p.exitcode is not None and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker {p.pid} crashed with exitcode {p.exitcode}! 已完成 {len(results)}/{total_expected} 盤。"
                        )
                
                # 🛡️ 死鎖偵測
                if alive_count > 0 and (time.time() - last_result_time) > STALL_TIMEOUT:
                    print(f"\n[WARNING] Self-play stall detected! {alive_count} workers alive but no results for {STALL_TIMEOUT}s. "
                          f"Collected {len(results)}/{total_expected}. Breaking out.", flush=True)
                    break
                if alive_count == 0:
                    break
                
            if dashboard:
                dashboard.status = f"[Phase A] 收集資料: 對局 {len(results)}/{total_expected} 完成..."
                try:
                    if server is not None and hasattr(server, 'gpu_mem_used'):
                        dashboard.gpu_mem_used = server.gpu_mem_used.value
                except Exception:
                    pass
                dashboard.update()

        for p in processes:
            p.join(timeout=30)
    finally:
        # 🛡️ 絕對防線：不管是正常結束還是 Ctrl+C 中斷，都必須強制終結所有子進程並釋放共享記憶體
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        server.stop()
        pool.cleanup()
        # 🛡️ 將模型重新搬回主進程指定的設備（通常為 GPU），以便接下來的訓練使用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

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
    
    # 🛡️ 設備一致性防禦：手動將優化器狀態 Tensor 移往與模型相同的設備，防止 CUDA/CPU 設備跨越崩潰
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # 🚀 優化：使用 torch.from_numpy().pin_memory() 進行記憶體釘鎖，並使用 non_blocking=True 非同步傳輸
    s = torch.from_numpy(states).pin_memory().to(device, non_blocking=True)
    pi = torch.from_numpy(target_probs).pin_memory().to(device, non_blocking=True)
    z = torch.from_numpy(target_values).pin_memory().to(device, non_blocking=True)
    t = torch.from_numpy(target_threats).pin_memory().to(device, non_blocking=True)  # [B, 15, 15] Aux Target

    optimizer.zero_grad()
    
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        log_p, v, aux = model(s, return_aux=True)  # 啟用 Aux Head
        # 🛡️ NaN 防禦：clamp log_p 防止 fp16 下溢產生 -inf
        log_p_safe = torch.clamp(log_p, min=-30.0)
        # AlphaZero 標準交叉熵
        policy_loss = -torch.sum(pi * log_p_safe) / pi.size(0)
        value_loss = F.mse_loss(v.view(-1), z.view(-1))
        # Aux Loss：讓 backbone 學會辨識棋型，權重 0.15
        aux_loss = F.mse_loss(aux, t)
        loss = policy_loss + 1.0 * value_loss + 0.15 * aux_loss

    # 🛡️ NaN 熔斷器：若 Loss 為 NaN/Inf，直接跳過此 step，保護模型權重
    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return {
            'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
            'aux_loss': 0.0, 'grad_norm': 0.0, 'policy_entropy': 0.0,
        }

    scaler.scale(loss).backward()
    
    # 📀 梯度範數監控 — 在 unscale 後計算
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    scaler.step(optimizer)
    scaler.update()

    # 📀 策略熵 (Policy Entropy) — 判斷策略是否崩潰
    with torch.no_grad():
        probs = torch.exp(log_p)  # log_softmax -> softmax
        entropy = -(probs * log_p).sum(dim=1).mean()

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
    move_count = 0

    while not env.done:
        if env.current_player == 1:
            mcts, predict_fn = black_mcts, black_predict
        else:
            mcts, predict_fn = white_mcts, white_predict

        # 🛡️ 破冰隨機性防禦：前 6 步使用高溫與 Dirichlet 噪聲，打破確定性黑棋必勝開局，防範平局死鎖
        if move_count < 6:
            temperature = 1.0
            dirichlet_alpha = 0.3
        else:
            temperature = 1e-3
            dirichlet_alpha = 0.0

        acts, probs = mcts.get_action_probs(
            env, predict_fn, temperature=temperature, dirichlet_alpha=dirichlet_alpha,
        )
        
        # 🛡️ 防禦：MCTS 回傳空結果 (終局邊界)
        if len(acts) == 0:
            break
        
        if move_count < 6:
            action = np.random.choice(acts, p=probs)
        else:
            action = acts[np.argmax(probs)]

        env.step(action)
        black_mcts.update_with_move(action)
        white_mcts.update_with_move(action)
        move_count += 1

    return env.winner


def _arena_single_game_worker(args):
    """
    並行 Arena 棋局進程的執行入口。
    """
    game_id, candidate_path, best_path, config, device_str, play_as_black = args
    
    # ⚙️ 限制 PyTorch 進程執行緒數量，防止 OpenMP/MKL 與 CUDA 競態死鎖
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device(device_str)
    
    # 載入候選模型 (最新訓練)
    candidate_model = PolicyValueNet().to(device)
    if os.path.exists(candidate_path):
        ckpt = torch.load(candidate_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        clean_state_dict = {
            (k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k): v
            for k, v in state_dict.items()
        }
        candidate_model.load_state_dict(clean_state_dict)
    candidate_model.eval()

    # 載入最佳模型 (擂主)
    best_model = PolicyValueNet().to(device)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        clean_state_dict = {
            (k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k): v
            for k, v in state_dict.items()
        }
        best_model.load_state_dict(clean_state_dict)
    best_model.eval()

    # ⚡ 使用 torch.jit.trace 優化 Arena 推論，防止 Windows 上 Triton 缺失導致的崩潰
    try:
        dummy_input = torch.zeros(1, 4, 15, 15).to(device)
        candidate_model = torch.jit.trace(candidate_model, dummy_input)
        best_model = torch.jit.trace(best_model, dummy_input)
    except Exception:
        pass

    candidate_predict = make_predict_fn(candidate_model, device)
    best_predict = make_predict_fn(best_model, device)

    if play_as_black:
        winner = play_arena_game(
            candidate_predict, best_predict,
            n_playout=config.arena_n_playout, c_puct=config.c_puct,
        )
    else:
        winner = play_arena_game(
            best_predict, candidate_predict,
            n_playout=config.arena_n_playout, c_puct=config.c_puct,
        )
    return winner


def arena_evaluate(candidate_path, best_path, config, device_str, dashboard=None):
    half = config.arena_games // 2
    candidate_wins = 0
    candidate_draws = 0

    log_file = "arena_process.log"

    # 建立任務參數列表
    tasks = []
    # 挑戰者執黑
    for i in range(half):
        tasks.append((i, candidate_path, best_path, config, device_str, True))
    # 挑戰者執白
    for i in range(half):
        tasks.append((half + i, candidate_path, best_path, config, device_str, False))

    # 使用 spawn 進程池並行執行對戰
    import multiprocessing as mp
    # ⚙️ 最佳化：將並行進程數提升至 4，以獲取兩倍 (200%) 的評估效能；
    # 同時保持 maxtasksperchild=1 設定，在每個任務結束後重啟進程以徹底釋放 CUDA Context 與顯存。
    pool_size = min(4, config.arena_games)
    
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Parallel Arena with {pool_size} workers...\n")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=pool_size, maxtasksperchild=1) as pool:
        # 非同步執行並收集結果
        results = pool.map(_arena_single_game_worker, tasks)

    # 統計勝率
    for idx, winner in enumerate(results):
        play_as_black = tasks[idx][5]
        game_num = tasks[idx][0] + 1
        half_val = half
        side = "Black" if play_as_black else "White"
        
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] play_arena_game (Cand as {side}) {game_num if play_as_black else game_num - half}/{half_val} done. Winner={winner}\n")
        
        if play_as_black:
            if winner == 1: candidate_wins += 1
            elif winner == 0: candidate_draws += 1
        else:
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


def arena_worker_process(candidate_path, best_path, config, device_str, result_queue):
    """
    Arena 背景進程：負責載入模型並進行並行對弈評估，不會阻塞主迴圈。
    """
    # ⚙️ 限制 PyTorch 進程執行緒數量，防止 OpenMP/MKL 與 CUDA 競態死鎖
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    log_file = "arena_process.log"
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Arena process started. PID={os.getpid()}\n")

    try:
        result = arena_evaluate(candidate_path, best_path, config, device_str, dashboard=None)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Parallel arena_evaluate completed successfully. Win rate: {result['win_rate']*100:.1f}%\n")
        result['candidate_path'] = candidate_path
        
        result_queue.put(result)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Result put to queue.\n")
    except Exception as e:
        import traceback
        crash_msg = f"Arena Process crashed:\n{traceback.format_exc()}"
        try:
            with open("arena_crash.log", "w", encoding="utf-8") as f:
                f.write(crash_msg)
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Arena Process crashed:\n{traceback.format_exc()}\n")
        except Exception:
            pass
        import sys
        print(crash_msg, file=sys.stderr, flush=True)
        os._exit(1)


# ==========================================
# Checkpoint
# ==========================================
def save_checkpoint(path, model, optimizer, scheduler, scaler, iteration, replay_buffer=None):
    # 🛡️ 剝離 torch.compile 的 _orig_mod 前綴，確保相容標準載入
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    temp_path = path + '.tmp'
    torch.save({
        'iteration': iteration,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
    }, temp_path)
    
    if os.path.exists(temp_path):
        if os.path.exists(path):
            os.remove(path)
        os.rename(temp_path, path)

    # 🚀 持久化經驗池，防範重啟資料斷崖與過擬合退化
    if replay_buffer is not None:
        buffer_dir = os.path.dirname(path)
        buffer_path = os.path.join(buffer_dir, 'replay_buffer.pkl')
        temp_buffer_path = buffer_path + '.tmp'
        try:
            import pickle
            with open(temp_buffer_path, 'wb') as f:
                pickle.dump({
                    'states': replay_buffer.states_buf[:replay_buffer.size],
                    'probs': replay_buffer.probs_buf[:replay_buffer.size],
                    'values': replay_buffer.values_buf[:replay_buffer.size],
                    'threats': replay_buffer.threats_buf[:replay_buffer.size],
                    'pos': replay_buffer.pos,
                    'size': replay_buffer.size
                }, f)
            if os.path.exists(temp_buffer_path):
                if os.path.exists(buffer_path):
                    os.remove(buffer_path)
                os.rename(temp_buffer_path, buffer_path)
        except Exception:
            if os.path.exists(temp_buffer_path):
                try:
                    os.remove(temp_buffer_path)
                except Exception:
                    pass


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device='cuda'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {
        (k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict, strict=False)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    elif scaler:
        try:
            device = scaler._device if hasattr(scaler, '_device') else 'cuda'
            scaler._scale = torch.tensor(256.0, device=device)
        except Exception:
            pass
    return checkpoint.get('iteration', 0)


# ==========================================
# Zombie Process Defense
# ==========================================
def clean_zombie_processes():
    if os.name == 'nt':
        import subprocess
        current_pid = os.getpid()
        # 尋找並強制關閉非當前 PID 且與 GomokuAlphaZero 訓練相關的殘留 python 進程
        cmd = f'powershell -Command "Get-CimInstance Win32_Process -Filter \\"name = \'python.exe\'\\" | Where-Object {{ $_.ProcessId -ne {current_pid} -and ($_.CommandLine -like \'*train_gomoku.py*\' -or $_.CommandLine -like \'*spawn_main*\') }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"'
        try:
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


# ==========================================
# Main Pipeline
# ==========================================
def main():
    clean_zombie_processes()
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 1. Dashboard
    dash = Dashboard(config.max_iterations)
    
    # 2. Model & AMP
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    latest_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')

    model = PolicyValueNet()  # 延遲 GPU 初始化，暫留 CPU 以防 spawn 死鎖
    best_model = PolicyValueNet()  # 永遠留在 CPU 進行 Arena 評估
    scaler = torch.amp.GradScaler(device='cuda')

    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=config.lr_min,
    )

    start_iteration = 0
    # 優先讀取 latest_model 恢復進度
    if os.path.exists(latest_path):
        start_iteration = load_checkpoint(
            latest_path, model, optimizer, scheduler, scaler, device='cpu'
        )
        dash.status = f"System Boot: Resumed from latest iteration {start_iteration}"
        if os.path.exists(best_model_path):
            best_model_checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            best_model.load_state_dict(best_model_checkpoint['model_state_dict'])
        else:
            best_model.load_state_dict(model.state_dict())
    elif os.path.exists(best_model_path):
        start_iteration = load_checkpoint(
            best_model_path, model, optimizer, scheduler, scaler, device='cpu'
        )
        best_model.load_state_dict(model.state_dict())
        dash.status = f"System Boot: Resumed from best iteration {start_iteration}"
    else:
        best_model.load_state_dict(model.state_dict())
        dash.status = "System Boot: Starting from scratch"

    # 🛡️ 學習率重載校正防禦：防止從舊存檔載入過高的學習率與退火週期
    if start_iteration > 0:
        config.learning_rate = 2e-4  # 強制初始學習率為優化值
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = config.learning_rate
        
        # 依據當前載入的輪次，基於新的 T_max=1000 重新計算正確的 Cosine 學習率
        import math
        T_cur = start_iteration
        T_max = 1000
        eta_min = config.lr_min
        eta_max = config.learning_rate
        new_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * min(T_cur, T_max) / T_max))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 同步更新 scheduler 的狀態
        scheduler.base_lrs = [config.learning_rate]
        scheduler.T_max = T_max
        scheduler.last_epoch = start_iteration
        
        print(f"[LR Correction] Reset base_lr={config.learning_rate}, calculated new_lr={new_lr:.7f}", flush=True)

    # 確保 best_model.pth 一定存在，防止背景 Arena 進程載入失敗
    if not os.path.exists(best_model_path):
        save_checkpoint(best_model_path, best_model, optimizer, scheduler, scaler, start_iteration)

    # 3. Buffer & TensorBoard
    replay_buffer = ReplayBuffer(max_size=config.buffer_max_size)
    buffer_path = os.path.join(config.checkpoint_dir, 'replay_buffer.pkl')
    if os.path.exists(buffer_path):
        try:
            import pickle
            dash.status = "Loading Replay Buffer from checkpoints..."
            dash.update()
            with open(buffer_path, 'rb') as f:
                buf_state = pickle.load(f)
                if 'buffer' in buf_state:
                    # 舊版格式相容載入
                    old_buf = buf_state['buffer']
                    replay_buffer.size = 0
                    replay_buffer.pos = 0
                    for item in old_buf:
                        p = replay_buffer.pos
                        replay_buffer.states_buf[p] = item[0]
                        replay_buffer.probs_buf[p] = item[1]
                        replay_buffer.values_buf[p] = item[2]
                        replay_buffer.threats_buf[p] = item[3]
                        replay_buffer.pos = (p + 1) % replay_buffer.max_size
                        replay_buffer.size = min(replay_buffer.size + 1, replay_buffer.max_size)
                else:
                    # 新版連續 NumPy 格式載入
                    saved_size = buf_state.get('size', 0)
                    replay_buffer.size = saved_size
                    replay_buffer.pos = buf_state.get('pos', 0)
                    if saved_size > 0:
                        replay_buffer.states_buf[:saved_size] = buf_state['states']
                        replay_buffer.probs_buf[:saved_size] = buf_state['probs']
                        replay_buffer.values_buf[:saved_size] = buf_state['values']
                        replay_buffer.threats_buf[:saved_size] = buf_state['threats']
            dash.status = f"Replay Buffer restored successfully: {len(replay_buffer):,} samples"
            dash.update()
            print(f"Replay Buffer loaded: {len(replay_buffer)} samples", flush=True)
            time.sleep(1.0)
        except Exception as e:
            dash.status = f"Warning: Failed to load Replay Buffer: {str(e)}"
            dash.update()
            print(f"Failed to load Replay Buffer: {e}", flush=True)
            time.sleep(1.5)
    writer = SummaryWriter(config.log_dir)
    print("TensorBoard SummaryWriter initialized.", flush=True)
    global_train_step = start_iteration * config.train_steps_per_iter

    # 4. Main Loop
    patience_counter = 0
    is_ice_breaking = False
    arena_process = None
    arena_result_queue = Queue()
    try:
        for iteration in range(start_iteration, config.max_iterations):
            iter_start = time.time()
            dash.iteration = iteration + 1
            
            # --- 曲率驅動動態更新 ---
            if config.max_iterations > 10:
                if iteration < 200:
                    # Phase A: 極速拓荒，瘋狂累積資料量
                    config.num_workers = 16
                    config.games_per_iteration = 48
                    config.n_playout = 200
                    config.train_steps_per_iter = 100
                    config.batch_size = 1024
                    config.arena_interval = 10
                    config.c_puct = 5.0
                    config.temp_threshold = 6
                    dash.phase_name = "Phase A: 快速拓荒 (16核海量對局・廣泛建構開局庫)"
                elif iteration < 800:
                    # Phase B: 平穩學習，逐漸加深戰術防守
                    config.num_workers = 16
                    config.games_per_iteration = 32
                    config.n_playout = 400
                    config.train_steps_per_iter = 120
                    config.batch_size = 1024
                    config.arena_interval = 5
                    config.c_puct = 4.0
                    config.temp_threshold = 6
                    dash.phase_name = "Phase B: 中盤拉扯 (16核深化・防守反擊與活三訓練)"
                elif iteration < 1600:
                    # Phase C: 精細戰略，降低 Worker 防止進程碎片化
                    config.num_workers = 12
                    config.games_per_iteration = 24
                    config.n_playout = 800
                    config.train_steps_per_iter = 250
                    config.batch_size = 1024
                    config.arena_interval = 3
                    config.c_puct = 3.0
                    config.temp_threshold = 6
                    dash.phase_name = "Phase C: 算力解放 (12核精算・深入推演 VCF 連殺)"
                else:
                    # Phase D: 絕對收斂期，Playout 1000 即可突破天花板
                    config.num_workers = 12
                    config.games_per_iteration = 24
                    config.n_playout = 1000
                    config.train_steps_per_iter = 250
                    config.batch_size = 1024
                    config.arena_interval = 3
                    config.c_puct = 3.0
                    config.temp_threshold = 5
                    dash.phase_name = "Phase D: 死神領域 (12核深算・防禦力極限打磨)"
            else:
                dash.phase_name = "Smoke Test Mode"

            # 破冰機制覆蓋
            if is_ice_breaking:
                config.dirichlet_alpha = 1.0
                config.temp_threshold = 12
            else:
                config.dirichlet_alpha = 0.08

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
            
            iter_start_time = time.time()
            dash.update()
            print(f"\n--- Starting Iteration {iteration + 1} ---", flush=True)

            # -- Phase A: Self-Play --
            model.eval()
            sp_start = time.time()
            print("Starting run_self_play...", flush=True)
            sp_stats = run_self_play(model, config, replay_buffer, dash)
            print("run_self_play finished successfully.", flush=True)
            sp_elapsed = time.time() - sp_start

            total_games = sp_stats['black_wins'] + sp_stats['white_wins'] + sp_stats['draws']
            dash.avg_moves = sp_stats['total_moves'] // max(1, total_games)
            dash.b_wins = sp_stats['black_wins']
            dash.w_wins = sp_stats['white_wins']
            dash.draws = sp_stats['draws']
            dash.buffer_size = len(replay_buffer)
            dash.draw_rate = dash.draws / max(1, total_games)
            dash.sp_speed = total_games / max(0.01, sp_elapsed / 60.0)

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
            # 測試模式下使用較小的 warmup，否則限制最少 50000 筆資料
            warmup_threshold = config.min_buffer_for_train if '--test' in sys.argv else max(config.min_buffer_for_train, 50000)
            if len(replay_buffer) >= warmup_threshold:
                model.to(device)  # 訓練前將模型移至 GPU
                epoch_losses = {'total': [], 'policy': [], 'value': [], 'aux': [],
                               'grad_norm': [], 'entropy': []}
                # 🛡️ 四階漸進式步數平滑鎖
                buffer_len = len(replay_buffer)
                max_buf = config.buffer_max_size
                if buffer_len < max_buf * 0.5:
                    current_steps = 50      # 第一階：極速安全暖身
                elif buffer_len < max_buf * 0.75:
                    current_steps = 100     # 第二階：適應中低量級資料
                elif buffer_len < max_buf * 0.9:
                    current_steps = 160     # 第三階：中盤深化
                else:
                    current_steps = config.train_steps_per_iter  # 第四階：恢復極限修煉
                for step in range(current_steps):
                    
                    if step % max(1, current_steps // 20) == 0:
                        dash.status = f"[Phase B] Training: 梯度下降 {step+1}/{current_steps}..."
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
                    # 📀 每步細粒度 Loss
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

                # 📀 每輪平均 Loss
                writer.add_scalar('Epoch/Total_Loss', dash.total_loss, iteration)
                writer.add_scalar('Epoch/Policy_Loss', dash.p_loss, iteration)
                writer.add_scalar('Epoch/Value_Loss', dash.v_loss, iteration)
                writer.add_scalar('Epoch/Aux_Loss', dash.aux_loss, iteration)
                writer.add_scalar('Epoch/GradNorm', dash.grad_norm, iteration)
                writer.add_scalar('Epoch/PolicyEntropy', dash.policy_entropy, iteration)
                writer.add_scalar('LR', dash.lr, iteration)
                
                # 訓練完即可存為最新版
                save_checkpoint(latest_path, model, optimizer, scheduler, scaler, iteration + 1, replay_buffer)
            else:
                dash.status = f"[Phase B] Data Mining (buffer {len(replay_buffer):,} < {warmup_threshold:,} warmup)"
                dash.update()
                # 🚀 拓荒期安全保護：即使不更新權重，也定時保存檢查點與最新經驗池
                save_checkpoint(latest_path, model, optimizer, scheduler, scaler, iteration + 1, replay_buffer)

            # -- Phase C: 非同步 Arena 結果檢查 --
            if arena_process is not None and not arena_process.is_alive():
                try:
                    arena_result = arena_result_queue.get(timeout=2.0)
                    has_result = True
                except Exception:
                    has_result = False

                if has_result:
                    wr = arena_result['win_rate']
                    dash.arena_last_wr = wr
                    
                    writer.add_scalar('Arena/Win_Rate', wr, iteration)
                    writer.add_scalar('Arena/Patience', patience_counter, iteration)

                    # 🛡️ 破冰期保護鎖：高噪聲探索期間自我對弈質量較低，嚴禁篡位，防止弱智化模型污染擂主
                    if is_ice_breaking:
                        dash.status = f"[{dash.phase_name[:7]}] 破冰中(禁用篡位)。勝率 {wr*100:.1f}%，回滾權重"
                        patience_counter += 1
                        model.load_state_dict(best_model.state_dict())
                        model.cpu()
                        if os.path.exists(best_model_path):
                            best_ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
                            if 'optimizer_state_dict' in best_ckpt:
                                optimizer.load_state_dict(best_ckpt['optimizer_state_dict'])
                            if 'scheduler_state_dict' in best_ckpt:
                                scheduler.load_state_dict(best_ckpt['scheduler_state_dict'])
                    elif wr >= config.win_rate_threshold:
                        is_vetoed = False
                        if wr <= 0.50 and (dash.p_loss > 1.2 or dash.total_loss > 1.5):
                            is_vetoed = True
                            dash.status = f"[{dash.phase_name[:7]}] >>> 檢測到模型數值嚴重退化 (p_loss {dash.p_loss:.4f})！一票否決平局篡位！<<<"
                            patience_counter += 1
                            model.load_state_dict(best_model.state_dict())
                            model.cpu()
                            if os.path.exists(best_model_path):
                                best_ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
                                if 'optimizer_state_dict' in best_ckpt:
                                    optimizer.load_state_dict(best_ckpt['optimizer_state_dict'])
                                if 'scheduler_state_dict' in best_ckpt:
                                    scheduler.load_state_dict(best_ckpt['scheduler_state_dict'])
                        
                        if not is_vetoed:
                            patience_counter = 0
                            is_ice_breaking = False
                            config.dirichlet_alpha = 0.3
                            
                            dash.best_win_rate = wr
                            dash.status = f"[{dash.phase_name[:7]}] >>> 新王誕生! 勝率 {wr*100:.1f}% 篡位成功! <<<"
                            
                            shutil.copyfile(arena_result['candidate_path'], best_model_path)
                            best_model.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=False)['model_state_dict'])
                            shutil.copyfile(best_model_path, os.path.join(config.checkpoint_dir, f'model_v{iteration + 1}.pth'))
                    else:
                        patience_counter += 1
                        dash.status = f"[{dash.phase_name[:7]}] >>> 踢館失敗 ({patience_counter}連敗). 勝率 {wr*100:.1f}%."
                        
                        # 🛡️ 溫和非同步回滾防禦
                        if patience_counter >= 3:
                            is_ice_breaking = True
                            dash.status = f"[{dash.phase_name[:7]}] 踢館 >=3 連敗! 啟動破冰(提升噪聲)並回滾權重!"
                            config.dirichlet_alpha = 1.0
                            
                            patience_counter = 0
                            model.load_state_dict(best_model.state_dict())
                            model.cpu()
                            if os.path.exists(best_model_path):
                                best_ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
                                if 'optimizer_state_dict' in best_ckpt:
                                    optimizer.load_state_dict(best_ckpt['optimizer_state_dict'])
                                if 'scheduler_state_dict' in best_ckpt:
                                    scheduler.load_state_dict(best_ckpt['scheduler_state_dict'])
                                
                                # 🛡️ 回滾學習率防禦校正：防止舊 checkpoints 將學習率覆載回 1e-3 垃圾狀態
                                for param_group in optimizer.param_groups:
                                    param_group['initial_lr'] = 2e-4
                                import math
                                T_cur = iteration + 1
                                T_max = 1000
                                eta_min = config.lr_min
                                eta_max = 2e-4
                                new_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * min(T_cur, T_max) / T_max))
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = new_lr
                                scheduler.base_lrs = [2e-4]
                                scheduler.T_max = T_max
                                scheduler.last_epoch = iteration + 1
                                print(f"[Rollback LR Defense] Reset base_lr=0.0002, calculated new_lr={new_lr:.7f}", flush=True)
                        
                    dash.arena_patience = patience_counter
                    writer.add_scalar('Arena/Best_Win_Rate', dash.best_win_rate, iteration)
                    
                    if os.path.exists(arena_result['candidate_path']):
                        os.remove(arena_result['candidate_path'])

                arena_process.join()
                arena_process = None
                dash.update()
                time.sleep(1.5)

            # -- Phase C: 觸發非同步 Arena --
            if len(replay_buffer) >= warmup_threshold and (iteration + 1) % config.arena_interval == 0:
                if arena_process is None:
                    cand_path = os.path.join(config.checkpoint_dir, f'cand_{iteration+1}.pth')
                    shutil.copyfile(latest_path, cand_path)
                    
                    dash.status = f"[{dash.phase_name[:7]}] 啟動背景 Arena... (主程序繼續訓練)"
                    dash.update()
                    
                    # 最佳化：在 cuda 設備上進行非同步 Arena 推論評估以獲得極限加速
                    arena_process = Process(
                        target=arena_worker_process,
                        args=(cand_path, best_model_path, config, 'cuda', arena_result_queue)
                    )
                    arena_process.start()
                else:
                    dash.status = f"[{dash.phase_name[:7]}] 上一輪 Arena 仍在進行中，跳過本次觸發。"
                    dash.update()
                
            # 👇 --- 新增：記憶體強制清道夫 --- 👇
            LOCAL_WORKER_CACHE.cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                dash.gpu_mem_used = torch.cuda.memory_allocated(device) / 1024**2
            # 👆 ---------------------------- 👆

            # ⭐ 記錄此輪耗時
            dash.iter_time = time.time() - iter_start_time
            writer.add_scalar('System/Iter_Time_Sec', dash.iter_time, iteration)
            if torch.cuda.is_available():
                writer.add_scalar('System/GPU_Memory_MB', dash.gpu_mem_used, iteration)
            writer.add_scalar('System/Buffer_Size', len(replay_buffer), iteration)

        # 測試模式下若有啟動背景 Arena，需在此等待其結束以確認結果與資源釋放
        if '--test' in sys.argv and arena_process is not None:
            dash.status = "Smoke Test: Waiting for Arena process..."
            dash.update()
            while arena_process.is_alive():
                try:
                    arena_result = arena_result_queue.get(timeout=1.0)
                    wr = arena_result['win_rate']
                    dash.arena_last_wr = wr
                    dash.status = f"Smoke Test Arena finished: win_rate={wr*100:.1f}%"
                except Exception:
                    pass
                dash.update()
            arena_process.join()

    except KeyboardInterrupt:
        if arena_process is not None and arena_process.is_alive():
            arena_process.terminate()
            arena_process.join()
        dash.status = "[!] USER INTERRUPT: 安全存檔中..."
        dash.update()
        try:
            save_iter = iteration + 1
        except NameError:
            save_iter = start_iteration
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, save_iter, replay_buffer)
        os._exit(0)
    finally:
        writer.close()
        dash.status = "[√] 訓練已安全結束！"
        dash.update()


if __name__ == '__main__':
    # 啟動日誌雙重記錄，主進程的所有輸出與 Exception 都將寫入實體檔案中
    sys.stdout = DualLogger("train_system.log", sys.stdout)
    sys.stderr = DualLogger("train_system.log", sys.stderr)
    sys.excepthook = exception_hook

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
