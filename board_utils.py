"""
board_utils.py - 五子棋棋盤運算共用核心
==========================================
所有 Numba JIT 加速的棋盤分析函式集中在此，
供 env.py、vcf_vct.py 等模組共用，避免循環 import。

包含：
  - _get_ray: 射線擷取
  - _analyze_ray: 棋型分析狀態機
  - _mock_play_check_overline_or_double_four: 長連/雙四虛擬檢測
  - _verify_suspect_three: 活三真偽驗證
  - _check_win_and_forbidden: 全局勝負/禁手判定
  - _generate_forbidden_mask: 全盤禁手遮罩生成
"""
import numpy as np
from numba import njit


# ==========================================
# ⚡ Numba C-Level 高速核心運算引擎
# ==========================================

@njit(fastmath=True, cache=True, nogil=True)
def _get_ray(board, x, y, dx, dy):
    """
    射線擷取：從 (x, y) 沿著 (dx, dy) 抓取長度為 9 的一維陣列 (-4 到 +4)。
    遇到邊界則填入 2 (代表牆壁/封閉端)。
    """
    ray = np.full(9, 2, dtype=np.int8)
    for i in range(-4, 5):
        nx, ny = x + i * dx, y + i * dy
        if 0 <= nx < 15 and 0 <= ny < 15:
            ray[i + 4] = board[nx, ny]
    return ray

@njit(fastmath=True, cache=True, nogil=True)
def _analyze_ray(ray, player_color):
    """
    極速自由度狀態機 (Window Sliding)
    針對長度 9 的射線，中心落子點必定在 ray[4]。
    回傳: (is_five, is_overline, suspect_threes, fours)
    """
    is_five = False
    is_overline = False
    suspect_threes = 0
    fours = 0
    
    # 1. 探測中心連續同色棋 (向左右延伸)
    left = 4
    while left > 0 and ray[left - 1] == player_color:
        left -= 1
    right = 4
    while right < 8 and ray[right + 1] == player_color:
        right += 1
        
    consecutive = right - left + 1
    
    if consecutive == 5:
        return True, False, 0, 0
    elif consecutive > 5:
        return False, True, 0, 0

    # 2. 判斷四 (Four) - 包括活四與衝四
    # 原理：包含 ray[4] 的長度為 5 的窗口內，有 4顆己方 + 1個空位
    for i in range(max(0, right - 4), min(5, left + 1)):
        stones = 0
        empty = 0
        for j in range(i, i + 5):
            if ray[j] == player_color:
                stones += 1
            elif ray[j] == 0:
                empty += 1
        if stones == 4 and empty == 1:
            fours += 1

    # 3. 判斷疑似活三 (Suspect Three)
    # 原理：包含 ray[4] 的長度為 6 的窗口內，有 3顆己方 + 3個空位
    # 這完美涵蓋了 011100, 001110, 以及跳三 010110 等自由度充足的活三！
    for i in range(max(0, right - 5), min(4, left + 1)):
        window_stones = 0
        window_empty = 0
        for j in range(i, i + 6):
            if ray[j] == player_color:
                window_stones += 1
            elif ray[j] == 0:
                window_empty += 1
        if window_stones == 3 and window_empty == 3 and ray[i] == 0 and ray[i + 5] == 0:
            suspect_threes += 1

    # 防重疊過濾：同一條線上即便匹配到多個窗口，對於整體交叉判定而言，
    # 只要具備四的威脅，這條線就算作 1 個 Four；具備活三潛力則為 1 個 Three。
    if fours > 0:
        fours = 1
        suspect_threes = 0 # 如果這條線已經成四，就不應該降級被算作三
    elif suspect_threes > 0:
        suspect_threes = 1

    return False, False, suspect_threes, fours

@njit(fastmath=True, cache=True, nogil=True)
def _mock_play_check_overline_or_double_four(board, mx, my):
    """
    Layer 2 輔助函式：在 (mx, my) 虛擬落子後，
    檢查該點是否觸發長連或雙四（不再遞迴檢查雙三，切斷無窮遞迴）。
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    total_fours = 0
    for dx, dy in directions:
        ray = _get_ray(board, mx, my, dx, dy)
        is_five, is_overline, s_threes, fours = _analyze_ray(ray, 1)  # 永遠檢查黑棋
        if is_overline:
            return True   # 長連 → 此點是禁手
        if is_five:
            return False  # 成五優先於任何禁手 → 此點安全
        total_fours += fours
    if total_fours >= 2:
        return True  # 雙四 → 此點是禁手
    return False

@njit(fastmath=True, cache=True, nogil=True)
def _verify_suspect_three(board, x, y, dx, dy, color):
    """
    Layer 2 Mock-Play 核心：驗證某條方向上的疑似活三是否為「真活三」。
    真活三 = 下一步能形成「活四」，且該落子點不是禁手（長連/雙四）。

    邏輯：在射線上找出能讓這條線變成「四」的空位，
    逐一虛擬落子，檢查是否觸發長連或雙四。
    如果所有能成四的空位都是禁手，這條三就是「假活三」。
    """
    ray = _get_ray(board, x, y, dx, dy)

    # 找出包含中心 (ray[4]) 的長度 5 窗口中，能形成四的空位
    candidate_offsets = []  # 相對於中心的偏移量
    for i in range(max(0, 4 - 4), min(5, 4 + 1)):  # 窗口起點 0~4
        stones = 0
        empty = 0
        empty_offset = -1
        for j in range(i, i + 5):
            if ray[j] == color:
                stones += 1
            elif ray[j] == 0:
                empty += 1
                empty_offset = j
        # 3 己方 + 2 空位 → 填入一顆後變成 4+1 = 四
        # 但我們要確保 ray[4] 在窗口內且是己方棋
        if stones == 3 and empty == 2:
            # 這個窗口有兩個空位，找出不是 ray[4] 位置的空位作為候選
            for j in range(i, i + 5):
                if ray[j] == 0 and j != 4:  # 排除中心本身(已落子)
                    candidate_offsets.append(j - 4)

    # 也檢查長度 5 中 4+1 的情況（直接延伸成四）
    for i in range(max(0, 4 - 4), min(5, 4 + 1)):
        stones = 0
        empty_j = -1
        for j in range(i, i + 5):
            if ray[j] == color:
                stones += 1
            elif ray[j] == 0:
                empty_j = j
        if stones == 4 and empty_j >= 0:
            candidate_offsets.append(empty_j - 4)

    if len(candidate_offsets) == 0:
        return False  # 無法成四 → 假活三

    # 對每個候選空位做虛擬落子
    for offset in candidate_offsets:
        mx = x + offset * dx
        my = y + offset * dy
        if 0 <= mx < 15 and 0 <= my < 15 and board[mx, my] == 0:
            board[mx, my] = 1  # 虛擬落子
            is_forbidden_at_mock = _mock_play_check_overline_or_double_four(board, mx, my)
            board[mx, my] = 0  # 復原
            if not is_forbidden_at_mock:
                return True  # 至少有一個安全的成四路徑 → 真活三

    return False  # 所有成四路徑都觸發禁手 → 假活三

@njit(fastmath=True, cache=True, nogil=True)
def _check_win_and_forbidden(board, x, y, color):
    """
    全局交叉判定：整合四條射線，判斷是否勝利，或是否踩中禁手(黑棋限定)。
    回傳：(is_win, is_forbidden)
    """
    # 四個軸向：水平、垂直、主對角、副對角
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    total_fours = 0
    total_suspect_threes = 0
    # 記錄哪些方向有疑似活三 (最多 4 條)
    three_dir_flags = np.zeros(4, dtype=np.int8)

    for d_idx in range(4):
        dx, dy = directions[d_idx]
        ray = _get_ray(board, x, y, dx, dy)
        is_five, is_overline, s_threes, fours = _analyze_ray(ray, color)

        # 只要有一條線湊滿5子且不是長連，直接勝利 (白棋長連也算贏)
        if color == 1:  # 黑棋
            if is_overline:
                return False, True  # 黑棋長連直接判禁手
            if is_five:
                return True, False
        else:  # 白棋
            if is_five or is_overline:
                return True, False

        total_fours += fours
        total_suspect_threes += s_threes
        if s_threes > 0:
            three_dir_flags[d_idx] = 1

    # ==========================================
    # 禁手判定 (僅黑棋)
    # ==========================================
    if color == 1:
        # 雙四直接判禁手 (不需要 Mock-Play)
        if total_fours >= 2:
            return False, True

        # 疑似雙三 → 觸發 Layer 2 Mock-Play 精密驗證
        if total_suspect_threes >= 2:
            verified_threes = 0
            for d_idx in range(4):
                if three_dir_flags[d_idx] == 1:
                    dx, dy = directions[d_idx]
                    if _verify_suspect_three(board, x, y, dx, dy, color):
                        verified_threes += 1
            if verified_threes >= 2:
                return False, True  # 真雙三禁手

    return False, False

@njit(fastmath=True, cache=True)
def _generate_forbidden_mask(board):
    """全盤掃描：生成禁手遮罩 (1=禁手, 0=合法)"""
    mask = np.zeros((15, 15), dtype=np.int8)
    for i in range(15):
        for j in range(15):
            if board[i, j] == 0:
                # 模擬黑棋下在此處
                board[i, j] = 1
                _, is_forbidden = _check_win_and_forbidden(board, i, j, 1)
                if is_forbidden:
                    mask[i, j] = 1
                board[i, j] = 0 # 復原
    return mask
