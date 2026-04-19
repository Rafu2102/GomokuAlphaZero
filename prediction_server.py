"""
prediction_server.py - Zero-Copy Dynamic Batching GPU Prediction Server
========================================================================
AlphaZero Gomoku 的中樞神經系統。
利用 OS 級共享記憶體實現 MCTS Worker 與 GPU 推論引擎之間的零拷貝通訊。

架構原理：
  Worker (CPU)                          Server (GPU)
  ┌─────────────┐                      ┌─────────────────┐
  │ MCTS 搜尋   │                      │ PyTorch Model   │
  │ ↓           │                      │ ↑               │
  │ 寫入 SHM    │ ──號碼牌──→          │ 收集 Batch      │
  │ Event.wait()│ ←──喚醒───           │ GPU 推論        │
  │ ↓           │                      │ 寫回 SHM        │
  │ 讀取結果    │                      │ Event.set()     │
  └─────────────┘                      └─────────────────┘
"""

import numpy as np
import torch
import atexit
import threading
import time
from multiprocessing import shared_memory, Array, Value, Condition, Event as MPEvent

# ==========================================
# 📐 全域常數
# ==========================================
BOARD_SIZE = 15
INPUT_CHANNELS = 4
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE  # 225


# ==========================================
# 🛡️ OS 級共享記憶體池 (附殭屍防禦)
# ==========================================
class SharedMemoryPool:
    """
    預先配置三大塊 OS 實體連續記憶體：inputs, policies, values。
    所有 Worker 透過指標偏移直接讀寫，完全繞過 Python 序列化。
    """

    def __init__(self, max_workers=12):
        self.max_workers = max_workers
        self._cleaned = False

        # 定義記憶體佈局
        self.in_shape = (max_workers, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        self.p_shape = (max_workers, ACTION_SIZE)
        self.v_shape = (max_workers, 1)

        in_bytes = int(np.prod(self.in_shape)) * 4   # float32 = 4 bytes
        p_bytes = int(np.prod(self.p_shape)) * 4
        v_bytes = int(np.prod(self.v_shape)) * 4

        # 殭屍防禦：先嘗試清除舊名記憶塊 (如果存在的狀況)
        for name in ['AlphaZero_IN', 'AlphaZero_P', 'AlphaZero_V']:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.unlink()
            except FileNotFoundError:
                pass

        # 宣告 OS 實體記憶體區塊
        self.shm_in = shared_memory.SharedMemory(create=True, size=in_bytes, name='AlphaZero_IN')
        self.shm_p = shared_memory.SharedMemory(create=True, size=p_bytes, name='AlphaZero_P')
        self.shm_v = shared_memory.SharedMemory(create=True, size=v_bytes, name='AlphaZero_V')

        # 映射為 C-Contiguous NumPy 視圖 (零拷貝)
        self.inputs = np.ndarray(self.in_shape, dtype=np.float32, buffer=self.shm_in.buf)
        self.policies = np.ndarray(self.p_shape, dtype=np.float32, buffer=self.shm_p.buf)
        self.values = np.ndarray(self.v_shape, dtype=np.float32, buffer=self.shm_v.buf)

        # 初始化歸零
        self.inputs.fill(0)
        self.policies.fill(0)
        self.values.fill(0)

        # 🚨 殭屍防禦：程式無論如何結束，都必須釋放記憶體
        atexit.register(self.cleanup)

    def get_shm_names(self):
        """回傳共享記憶體名稱，供子進程 attach"""
        return (self.shm_in.name, self.shm_p.name, self.shm_v.name)

    def cleanup(self):
        """強制釋放 OS 實體記憶體，防止殭屍佔用"""
        if self._cleaned:
            return
        self._cleaned = True
        for shm in [self.shm_in, self.shm_p, self.shm_v]:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


# ==========================================
# ⚡ GPU 動態批次推論引擎 (Zero-Spin Consumer)
# ==========================================
class PredictionServer:
    """
    GPU Dynamic Batching Inference Server.

    核心同步機制：
    - Condition 鎖 + Array + Value = 單一真理來源的號碼牌系統
    - Event 陣列 = 精準定點喚醒，不做廣播
    - wait_for(timeout=0.002) = 零空轉動態超時
    """

    def __init__(self, model, pool, max_workers=12, batch_timeout_sec=0.002):
        self.model = model
        self.pool = pool
        self.max_workers = max_workers
        self.batch_timeout = batch_timeout_sec
        self._running = False

        # 同步原語：全部被同一把 Condition 鎖保護
        self.condition = Condition()
        self.pending_slots = Array('i', max_workers)   # 號碼牌陣列
        self.pending_count = Value('i', 0)              # 目前排隊人數
        self.worker_events = [MPEvent() for _ in range(max_workers)]  # 精準喚醒器

        # GPU 配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # 推論統計
        self._total_inferences = 0
        self._total_batches = 0

    def start(self):
        """在獨立守護線程中啟動推論引擎"""
        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """安全關閉推論引擎"""
        self._running = False
        with self.condition:
            self.condition.notify_all()
        if hasattr(self, '_thread'):
            self._thread.join(timeout=5)

    def _serve_loop(self):
        """
        主推論迴圈：等待 → 收集 → GPU 推論 → 寫回 → 精準喚醒
        """
        # print(f"[PredictionServer] 上線 | Device: {self.device} | Max Workers: {self.max_workers}")

        try:
            while self._running:
                # ── 等待階段 (Zero-Spin) ──
                with self.condition:
                    # wait_for + timeout：滿載立即出發，否則最多等 2ms
                    self.condition.wait_for(
                        lambda: self.pending_count.value > 0 or not self._running,
                        timeout=self.batch_timeout
                    )

                    count = self.pending_count.value
                    if count == 0:
                        continue

                    # 快照並清空號碼牌 (在鎖內完成，保證原子性)
                    batch_indices = [self.pending_slots[i] for i in range(count)]
                    self.pending_count.value = 0

                # ── GPU 推論階段 (鎖外執行，不阻塞新提交) ──
                # 從共享記憶體切片取出 Batch，送入顯卡
                batch_np = self.pool.inputs[batch_indices]  # fancy index = 自動拷貝
                batch_tensor = torch.from_numpy(batch_np).to(self.device)

                with torch.no_grad():
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        log_probs, values = self.model(batch_tensor)
                    probs = torch.exp(log_probs)

                # 寫回共享記憶體對應位置
                probs_np = probs.cpu().numpy()
                values_np = values.cpu().numpy()

                for i, idx in enumerate(batch_indices):
                    self.pool.policies[idx] = probs_np[i]
                    self.pool.values[idx] = values_np[i]

                # 更新統計
                self._total_batches += 1
                self._total_inferences += count

                # ── 精準喚醒 (Targeted Wake-up) ──
                for idx in batch_indices:
                    self.worker_events[idx].set()

        except Exception as e:
            print(f"[PredictionServer] 致命錯誤: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            avg_batch = (self._total_inferences / max(1, self._total_batches))
            # print(f"[PredictionServer] 已關閉 | "
            #       f"總推論: {self._total_inferences} | "
            #       f"總批次: {self._total_batches} | "
            #       f"平均批次大小: {avg_batch:.1f}")

    def get_sync_primitives(self):
        """回傳同步原語，供 PredictionClient 在子進程中使用"""
        return {
            'condition': self.condition,
            'pending_slots': self.pending_slots,
            'pending_count': self.pending_count,
            'worker_events': self.worker_events,
        }


# ==========================================
# 🎫 輕量級跨進程預測客戶端 (Picklable Producer)
# ==========================================
class PredictionClient:
    """
    Worker 端的預測介面。不包含任何 Thread 或 CUDA 模型，
    只持有可序列化的同步原語與共享記憶體名稱，
    因此能安全地透過 multiprocessing.Process 傳遞。
    """

    def __init__(self, worker_id, shm_names, condition, pending_slots,
                 pending_count, worker_event):
        self.worker_id = worker_id
        self.condition = condition
        self.pending_slots = pending_slots
        self.pending_count = pending_count
        self.worker_event = worker_event

        # Attach 到父進程建立的共享記憶體 (只讀寫自己的 Slot)
        in_name, p_name, v_name = shm_names
        self._shm_in = shared_memory.SharedMemory(name=in_name, create=False)
        self._shm_p = shared_memory.SharedMemory(name=p_name, create=False)
        self._shm_v = shared_memory.SharedMemory(name=v_name, create=False)

        max_workers = len(pending_slots)
        in_shape = (max_workers, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        p_shape = (max_workers, ACTION_SIZE)
        v_shape = (max_workers, 1)

        self.inputs = np.ndarray(in_shape, dtype=np.float32, buffer=self._shm_in.buf)
        self.policies = np.ndarray(p_shape, dtype=np.float32, buffer=self._shm_p.buf)
        self.values = np.ndarray(v_shape, dtype=np.float32, buffer=self._shm_v.buf)

    def predict(self, state_tensor):
        """
        提交預測請求並阻塞至 GPU 推論完成。

        Args:
            state_tensor: shape (4, 15, 15) 的 float32 NumPy 陣列
        Returns:
            (policy, value): policy 為 (225,) 機率向量，value 為 float
        """
        wid = self.worker_id

        # 1. 寫入共享記憶體 (C-Level 指標覆寫)
        self.inputs[wid] = state_tensor

        # 2. 清除前一輪的喚醒信號
        self.worker_event.clear()

        # 3. 提交號碼牌 (單一 Condition 鎖保護)
        with self.condition:
            slot = self.pending_count.value
            self.pending_slots[slot] = wid
            self.pending_count.value = slot + 1
            self.condition.notify()

        # 4. OS 級休眠 (0% CPU)
        self.worker_event.wait()

        # 5. 讀取結果
        policy = self.policies[wid].copy()
        value = float(self.values[wid][0])
        return policy, value

    def predict_for_mcts(self, state_tensor):
        """
        MCTS 轉接器：將 (225,) 扁平陣列轉譯為 MCTS 看得懂的
        [(action, prob), ...] 格式。
        """
        policy, value = self.predict(state_tensor)
        action_probs = list(enumerate(policy))
        return action_probs, value

    def close(self):
        """釋放子進程端的共享記憶體 Handle (不 unlink)"""
        for shm in [self._shm_in, self._shm_p, self._shm_v]:
            try:
                shm.close()
            except Exception:
                pass


# ==========================================
# 🧪 壓力測試 (Thread-Simulated Workers + PredictionClient)
# ==========================================
if __name__ == '__main__':
    from resnet import PolicyValueNet

    print("=" * 64)
    print("  Zero-Copy Prediction Server + Client 壓力測試")
    print("=" * 64)

    NUM_WORKERS = 4
    NUM_ROUNDS = 3

    # 1. 建立共享記憶體池
    pool = SharedMemoryPool(max_workers=NUM_WORKERS)
    print(f"[OK] 共享記憶體池建立完成")
    print(f"    Input  SHM: {pool.shm_in.name} ({pool.inputs.nbytes:,} bytes)")
    print(f"    Policy SHM: {pool.shm_p.name} ({pool.policies.nbytes:,} bytes)")
    print(f"    Value  SHM: {pool.shm_v.name} ({pool.values.nbytes:,} bytes)")

    # 2. 建立模型與推論引擎
    model = PolicyValueNet()
    server = PredictionServer(model, pool, max_workers=NUM_WORKERS)
    server.start()
    time.sleep(0.5)

    # 3. 取得同步原語，建立 Client 陣列
    sync = server.get_sync_primitives()
    shm_names = pool.get_shm_names()

    clients = []
    for i in range(NUM_WORKERS):
        c = PredictionClient(
            worker_id=i,
            shm_names=shm_names,
            condition=sync['condition'],
            pending_slots=sync['pending_slots'],
            pending_count=sync['pending_count'],
            worker_event=sync['worker_events'][i],
        )
        clients.append(c)

    # 4. 模擬 Worker 併發提交
    results = {}
    errors = []

    def simulate_worker(client, rounds):
        try:
            for r in range(rounds):
                fake_state = np.random.randn(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
                policy, value = client.predict(fake_state)
                results[(client.worker_id, r)] = (policy.shape, value)
        except Exception as e:
            errors.append((client.worker_id, str(e)))

    threads = []
    t_start = time.perf_counter()
    for c in clients:
        t = threading.Thread(target=simulate_worker, args=(c, NUM_ROUNDS))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.perf_counter() - t_start

    # 5. 驗證
    print(f"\n{'---' * 21}")
    print(f"  測試結果 ({NUM_WORKERS} Workers x {NUM_ROUNDS} Rounds)")
    print(f"{'---' * 21}")
    all_ok = True
    for wid in range(NUM_WORKERS):
        for r in range(NUM_ROUNDS):
            key = (wid, r)
            if key not in results:
                print(f"  [FAIL] Worker {wid} Round {r}: timeout")
                all_ok = False
            else:
                p_shape, v = results[key]
                ok = p_shape == (ACTION_SIZE,) and -1.0 <= v <= 1.0
                tag = "OK" if ok else "FAIL"
                print(f"  [{tag}] Worker {wid} Round {r}: Policy={p_shape}, Value={v:+.4f}")
                if not ok:
                    all_ok = False
    for wid, err in errors:
        print(f"  [FAIL] Worker {wid}: {err}")
        all_ok = False

    total_inferences = NUM_WORKERS * NUM_ROUNDS
    print(f"\n  Throughput: {total_inferences / elapsed:.1f} inferences/sec ({elapsed:.4f}s)")

    # 6. 清理
    for c in clients:
        c.close()
    server.stop()
    pool.cleanup()

    print(f"\n{'=' * 64}")
    print(f"  {'ALL TESTS PASSED' if all_ok else 'TESTS FAILED'}")
    print(f"{'=' * 64}")
