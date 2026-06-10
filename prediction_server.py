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
import atexit
import threading
import time
import os
from multiprocessing import shared_memory, Array, Value, Condition, Event as MPEvent, Queue as MPQueue, Process

# ==========================================
# 🔐 執行緒安全的共享記憶體命名計數器
# ==========================================
_shm_counter = 0
_shm_counter_lock = threading.Lock()

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
        self.s_shape = (max_workers,)

        in_bytes = int(np.prod(self.in_shape)) * 4   # float32 = 4 bytes
        p_bytes = int(np.prod(self.p_shape)) * 4
        v_bytes = int(np.prod(self.v_shape)) * 4
        s_bytes = int(np.prod(self.s_shape)) * 1     # int8 = 1 byte

        # 🛡️ 絕對唯一命名防禦：結合 PID、高精度時間戳記、遞增計數器與隨機後綴，徹底杜絕 Windows 殭屍記憶體同名衝突
        global _shm_counter
        with _shm_counter_lock:
            _shm_counter += 1
            count = _shm_counter
        pid = os.getpid()
        t_ns = time.time_ns()
        import uuid
        rand_suffix = uuid.uuid4().hex[:4]
        uid = f"{pid}_{t_ns}_{count}_{rand_suffix}"
        
        self.in_name = f"AlphaZero_IN_{uid}"
        self.p_name = f"AlphaZero_P_{uid}"
        self.v_name = f"AlphaZero_V_{uid}"
        self.s_name = f"AlphaZero_S_{uid}"

        # 宣告 OS 實體記憶體區塊，帶有大小檢查與殭屍清理之強健防護
        self.shm_in = self._create_or_attach_shm(self.in_name, in_bytes)
        self.shm_p = self._create_or_attach_shm(self.p_name, p_bytes)
        self.shm_v = self._create_or_attach_shm(self.v_name, v_bytes)
        self.shm_s = self._create_or_attach_shm(self.s_name, s_bytes)

        # 映射為 C-Contiguous NumPy 視圖（零拷貝）
        self.inputs = np.ndarray(self.in_shape, dtype=np.float32, buffer=self.shm_in.buf)
        self.policies = np.ndarray(self.p_shape, dtype=np.float32, buffer=self.shm_p.buf)
        self.values = np.ndarray(self.v_shape, dtype=np.float32, buffer=self.shm_v.buf)
        self.states = np.ndarray(self.s_shape, dtype=np.int8, buffer=self.shm_s.buf)

        # 初始化歸零
        self.inputs.fill(0)
        self.policies.fill(0)
        self.values.fill(0)
        self.states.fill(0)

        # 🚨 殭屍防禦：程式無論如何結束，都必須釋放記憶體
        atexit.register(self.cleanup)

    def _create_or_attach_shm(self, name, size):
        try:
            return shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            # 殭屍防禦：若已被佔用，則 attach 至現有區塊並檢查大小是否符合
            shm = shared_memory.SharedMemory(create=False, name=name)
            if shm.size >= size:
                return shm
            else:
                # 大小不足，判定為舊有的殘留無效共享記憶體，強制清理並重新建立
                shm.close()
                try:
                    shm.unlink()
                except Exception:
                    pass
                return shared_memory.SharedMemory(create=True, size=size, name=name)

    def get_shm_names(self):
        """回傳共享記憶體名稱，供子進程 attach"""
        return (self.shm_in.name, self.shm_p.name, self.shm_v.name, self.shm_s.name)

    def cleanup(self):
        """強制釋放 OS 實體記憶體，防止殭屍佔用"""
        if self._cleaned:
            return
        self._cleaned = True
        for shm in [self.shm_in, self.shm_p, self.shm_v, self.shm_s]:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


# ==========================================
# ⚡ GPU 動態批次推論引擎 (Zero-Spin Consumer)
# ==========================================
class PredictionServer(Process):
    """
    GPU Dynamic Batching Inference Server (獨立進程版).
    """

    def __init__(self, model, pool, max_workers=12, batch_timeout_sec=0.002):
        super().__init__()
        self.model = model
        self.pool = pool
        self.max_workers = max_workers
        self.batch_timeout = batch_timeout_sec
        self._running = Value('b', True)

        # 🛡️ 防禦性斷言：確保傳入的模型在 CPU 上，防止 Windows spawn 序列化崩潰
        for param in model.parameters():
            if param.device.type == 'cuda':
                raise RuntimeError("致命錯誤：傳入 PredictionServer 的模型仍留在 GPU 上！請先呼叫 model.cpu()")

        # 同步原語：使用 Queue 排隊，Event 喚醒
        self.request_queue = MPQueue()
        self.worker_events = [MPEvent() for _ in range(max_workers)]  # 精準喚醒器

        # 推論統計
        self._total_inferences = 0
        self._total_batches = 0
        
        # ⚡ 共享顯存監控：供主進程 Dashboard 讀取子進程的真實 GPU 顯存
        self.gpu_mem_used = Value('f', 0.0)

    def run(self):
        """進程主入口：在獨立子進程中安全初始化 CUDA 並執行推論"""
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        # ⚡ 使用 torch.jit.trace 優化 eval 模型，防止 Windows 上 Triton 缺失導致的崩潰
        try:
            dummy_input = torch.zeros(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).to(self.device)
            self.model = torch.jit.trace(self.model, dummy_input)
        except Exception as e:
            print(f"[PredictionServer] JIT trace 失敗: {e}，回退至 eager 模式")

        # ⚡ 寫入初始顯存
        if self.device.type == 'cuda':
            self.gpu_mem_used.value = float(torch.cuda.memory_allocated(self.device) / 1024**2)
        self._serve_loop()

    def stop(self):
        """安全關閉推論引擎進程"""
        self._running.value = False
        self.join(timeout=5)
        if self.is_alive():
            self.terminate()
            self.join()

    def _serve_loop(self):
        """
        優化後的主推論迴圈：增加極短暫延遲以凝聚更大的 Batch，發揮 GPU 威力
        """
        import torch
        from multiprocessing import shared_memory
        shm_in = shared_memory.SharedMemory(name=self.pool.in_name, create=False)
        shm_p = shared_memory.SharedMemory(name=self.pool.p_name, create=False)
        shm_v = shared_memory.SharedMemory(name=self.pool.v_name, create=False)
        shm_s = shared_memory.SharedMemory(name=self.pool.s_name, create=False)

        inputs_view = np.ndarray(self.pool.in_shape, dtype=np.float32, buffer=shm_in.buf)
        policies_view = np.ndarray(self.pool.p_shape, dtype=np.float32, buffer=shm_p.buf)
        values_view = np.ndarray(self.pool.v_shape, dtype=np.float32, buffer=shm_v.buf)
        states_view = np.ndarray(self.pool.s_shape, dtype=np.int8, buffer=shm_s.buf)

        try:
            from queue import Empty
            while self._running.value:
                # ── 等待與收集階段 (Zero-Spin Dynamic Batching) ──
                try:
                    # 阻塞等待第一個請求 (Timeout 防止停止時卡死)
                    _ = self.request_queue.get(timeout=0.1)
                except Empty:
                    continue

                if not self._running.value:
                    break

                # ⚡ 關鍵優化：稍微等待 0.2 毫秒，讓其他同時在搜 MCTS 的 Worker 有時間把資料寫進 shm
                # 這能讓平均 Batch Size 從 1~2 瞬間提升到 8~12，GPU 利用率暴增
                time.sleep(0.0002)

                # 🚀 革命性優化：直接從共享記憶體掃描所有處於 REQUESTED (1) 狀態的 Workers
                batch_indices = []
                for wid in range(self.max_workers):
                    if states_view[wid] == 1:
                        states_view[wid] = 2  # 標記為 PROCESSING (2)
                        batch_indices.append(wid)

                if not batch_indices:
                    continue

                count = len(batch_indices)

                # ── GPU 推論階段 (鎖外執行，不阻塞新提交) ──
                batch_np = inputs_view[batch_indices]  # fancy index = 自動拷貝
                batch_tensor = torch.from_numpy(batch_np).to(self.device)

                with torch.no_grad():
                    if self.device.type == 'cpu':
                        log_probs, values = self.model(batch_tensor)
                    else:
                        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                            log_probs, values = self.model(batch_tensor)
                    probs = torch.exp(log_probs)

                # 寫回共享記憶體對應位置
                probs_np = probs.cpu().numpy()
                values_np = values.cpu().numpy()

                for i, idx in enumerate(batch_indices):
                    policies_view[idx] = probs_np[i]
                    values_view[idx] = values_np[i]
                    states_view[idx] = 3  # 標記為 READY (3)

                # 更新統計
                self._total_batches += 1
                self._total_inferences += count

                # ⚡ 定期更新共享的顯存佔用 (每 50 個 batches 統計一次，以兼顧效能)
                if self._total_batches % 50 == 0 and self.device.type == 'cuda':
                    self.gpu_mem_used.value = float(torch.cuda.memory_allocated(self.device) / 1024**2)

                # ── 精準喚醒 (Targeted Wake-up) ──
                for idx in batch_indices:
                    self.worker_events[idx].set()

        except Exception as e:
            print(f"[PredictionServer] 致命錯誤: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            # 🚀 釋放子進程端的共享記憶體 handle
            for shm in [shm_in, shm_p, shm_v, shm_s]:
                try:
                    shm.close()
                except Exception:
                    pass


    def get_sync_primitives(self):
        """回傳同步原語，供 PredictionClient 在子進程中使用"""
        return {
            'request_queue': self.request_queue,
            'worker_events': self.worker_events,
        }


# ==========================================
# 🎫 輕量級跨進程預測客戶端 (Picklable Producer)
# ==========================================
class PredictionClient:
    """
    Worker 端的預測介面。不包含 any Thread 或 CUDA 模型，
    只持有可序列化的同步原語與共享記憶體名稱，
    因此能安全地透過 multiprocessing.Process 傳遞。
    """

    def __init__(self, worker_id, shm_names, request_queue, worker_event, max_workers):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.worker_event = worker_event

        # Attach 到父進程建立的共享記憶體 (只讀寫自己的 Slot)
        in_name, p_name, v_name, s_name = shm_names
        self._shm_in = shared_memory.SharedMemory(name=in_name, create=False)
        self._shm_p = shared_memory.SharedMemory(name=p_name, create=False)
        self._shm_v = shared_memory.SharedMemory(name=v_name, create=False)
        self._shm_s = shared_memory.SharedMemory(name=s_name, create=False)

        in_shape = (max_workers, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        p_shape = (max_workers, ACTION_SIZE)
        v_shape = (max_workers, 1)
        s_shape = (max_workers,)

        self.inputs = np.ndarray(in_shape, dtype=np.float32, buffer=self._shm_in.buf)
        self.policies = np.ndarray(p_shape, dtype=np.float32, buffer=self._shm_p.buf)
        self.values = np.ndarray(v_shape, dtype=np.float32, buffer=self._shm_v.buf)
        self.states = np.ndarray(s_shape, dtype=np.int8, buffer=self._shm_s.buf)

    def predict(self, state_tensor):
        """
        提交預測請求並阻塞至 GPU 推論完成。
        內建防死鎖機制：若 Event 信號因競態條件遺失，自動重新提交。

        Args:
            state_tensor: shape (4, 15, 15) 的 float32 NumPy 陣列
        Returns:
            (policy, value): policy 為 (225,) 機率向量，value 為 float
        """
        wid = self.worker_id
        MAX_RETRIES = 5
        WAIT_TIMEOUT = 60.0  # 秒 (給予首輪編譯暖身充足時間)

        for attempt in range(MAX_RETRIES):
            # 1. 寫入共享記憶體 (C-Level 指標覆寫)
            self.inputs[wid] = state_tensor

            # 2. 清空 Event 並設定狀態為 REQUESTED (1)
            self.worker_event.clear()
            self.states[wid] = 1

            # 3. 提交請求號碼牌以喚醒 Server
            self.request_queue.put(wid)

            # 4. OS 級休眠，帶超時防死鎖
            signaled = self.worker_event.wait(timeout=WAIT_TIMEOUT)
            if self.states[wid] == 3:
                break
            if signaled:
                break
            # Event 超時 → 競態條件導致信號遺失，重新提交

        # 5. 讀取結果
        policy = self.policies[wid].copy()
        value = float(self.values[wid][0])
        # 6. 重置狀態為 IDLE (0)
        self.states[wid] = 0
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
        for shm in [self._shm_in, self._shm_p, self._shm_v, self._shm_s]:
            try:
                shm.close()
            except Exception:
                pass
