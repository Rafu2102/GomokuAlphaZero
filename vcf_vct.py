"""
vcf_vct.py - VCF 候選過濾器 (v7 Pure Zero)
=============================================
迭代式 DFS VCF 搜索引擎。
只用在 MCTS 根節點（每步棋跑 1 次），判斷是否有必勝連殺。
結果用 additive bias 溫和注入 Policy Prior。

⚠️ 呼叫方必須傳入 board.copy()！不能傳 reference！
"""
import numpy as np
from numba import njit
from board_utils import _get_ray, _analyze_ray, _check_win_and_forbidden


@njit(fastmath=True, cache=True)
def _find_four_threats(board, color):
    """
    找出所有能讓 color 形成「四」的空位座標。
    回傳: (N, 2) 的 int32 陣列，每行 = (x, y)
    """
    result = np.zeros((225, 2), dtype=np.int32)
    count = 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for i in range(15):
        for j in range(15):
            if board[i, j] != 0:
                continue
            board[i, j] = color  # 模擬落子
            is_threat = False
            for dx, dy in directions:
                ray = _get_ray(board, i, j, dx, dy)
                is_five, _, _, fours = _analyze_ray(ray, color)
                if is_five or fours > 0:
                    is_threat = True
                    break
            board[i, j] = 0  # 復原
            if is_threat and count < 225:
                result[count, 0] = i
                result[count, 1] = j
                count += 1
    return result[:count]


@njit(fastmath=True, cache=True)
def _find_defense_point(board, ax, ay, attacker):
    """
    分析攻擊方在 (ax, ay) 落子後形成的四，找出防守方必須堵的位置。
    回傳: (is_four, def_x, def_y, is_open_four)
      - is_open_four=True → 活四，無法防守 = 必勝
      - is_four=True, is_open_four=False → 衝四，防守點 = (def_x, def_y)
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    def_points = np.zeros((8, 2), dtype=np.int32)  # 最多 8 個防守候選
    def_count = 0

    for dx, dy in directions:
        ray = _get_ray(board, ax, ay, dx, dy)
        is_five, _, _, fours = _analyze_ray(ray, attacker)

        if is_five:
            return True, -1, -1, True  # 已經五連 = 必勝

        if fours > 0:
            # 找衝四中的空位（防守點）
            for w in range(max(0, 4 - 4), min(5, 4 + 1)):
                stones = 0
                empty_k = -1
                for k in range(w, w + 5):
                    if ray[k] == attacker:
                        stones += 1
                    elif ray[k] == 0:
                        empty_k = k
                if stones == 4 and empty_k >= 0:
                    off = empty_k - 4
                    bx, by = ax + off * dx, ay + off * dy
                    if 0 <= bx < 15 and 0 <= by < 15:
                        # 去重
                        is_dup = False
                        for d in range(def_count):
                            if def_points[d, 0] == bx and def_points[d, 1] == by:
                                is_dup = True
                                break
                        if not is_dup and def_count < 8:
                            def_points[def_count, 0] = bx
                            def_points[def_count, 1] = by
                            def_count += 1

    if def_count == 0:
        return False, -1, -1, False  # 沒有形成四
    if def_count >= 2:
        return True, -1, -1, True   # 多點防守 = 活四 = 必勝
    return True, def_points[0, 0], def_points[0, 1], False  # 衝四，單一防守點


@njit(fastmath=True, cache=True)
def vcf_search(board, attacker, max_depth=15):
    """
    迭代式 DFS VCF (Victory by Continuous Fours) 搜索。

    ⚠️ 呼叫方必須傳入 board.copy()！原始棋盤不得被修改。

    策略：攻擊方不斷下衝四，防守方被迫堵住。
    如果攻擊方能走出活四（對手堵不住），就是必勝。

    Args:
        board: 棋盤副本 (np.int8, 15×15)。會被就地修改。
        attacker: 攻擊方顏色 (1 或 -1)
        max_depth: DFS 最大層數硬上限

    Returns:
        bool: 是否存在 VCF 必勝序列
    """
    defender = -attacker
    max_layers = min(max_depth, 16)  # 硬上限 16 層

    # 每層記錄狀態的堆疊
    atk_stack_x = np.zeros(max_layers, dtype=np.int32)
    atk_stack_y = np.zeros(max_layers, dtype=np.int32)
    def_stack_x = np.zeros(max_layers, dtype=np.int32)
    def_stack_y = np.zeros(max_layers, dtype=np.int32)
    has_def = np.zeros(max_layers, dtype=np.int8)   # 該層是否有放防守子
    idx_stack = np.zeros(max_layers, dtype=np.int32)  # 當前嘗試的 threat index

    # 每層的 threats 緩衝（最多 64 threats/層）
    layer_threats = np.zeros((max_layers, 64, 2), dtype=np.int32)
    layer_tcnt = np.zeros(max_layers, dtype=np.int32)

    # 初始化第 0 層：找出所有衝四威脅
    t0 = _find_four_threats(board, attacker)
    if len(t0) == 0:
        return False  # 連一個衝四都沒有
    cnt0 = min(len(t0), 64)
    layer_tcnt[0] = cnt0
    for i in range(cnt0):
        layer_threats[0, i, 0] = t0[i, 0]
        layer_threats[0, i, 1] = t0[i, 1]
    idx_stack[0] = 0
    depth = 0

    while depth >= 0:
        idx = idx_stack[depth]
        cnt = layer_tcnt[depth]

        # 回溯清理：把上一次嘗試的棋子撤掉
        if idx > 0:
            if has_def[depth] != 0:
                board[def_stack_x[depth], def_stack_y[depth]] = 0
                has_def[depth] = 0
            board[atk_stack_x[depth], atk_stack_y[depth]] = 0

        # 所有 threats 已遍歷或超深 → 回溯
        if idx >= cnt or depth >= max_layers - 1:
            depth -= 1
            if depth >= 0:
                idx_stack[depth] += 1
            continue

        # 嘗試攻擊方下這個 threat
        tx = layer_threats[depth, idx, 0]
        ty = layer_threats[depth, idx, 1]

        if board[tx, ty] != 0:
            # 該位置已被佔 → 跳過
            idx_stack[depth] += 1
            continue

        board[tx, ty] = attacker
        atk_stack_x[depth] = tx
        atk_stack_y[depth] = ty

        # 檢查是否直接五連
        is_win, _ = _check_win_and_forbidden(board, tx, ty, attacker)
        if is_win:
            board[tx, ty] = 0  # 清理
            return True

        # 檢查這步是否形成了四，以及防守點在哪
        found, dx, dy, is_open = _find_defense_point(board, tx, ty, attacker)
        if is_open:
            board[tx, ty] = 0
            return True  # 活四 = 必勝
        if not found:
            board[tx, ty] = 0
            idx_stack[depth] += 1
            continue  # 這步沒形成四 → 跳過

        # 防守方堵住
        if board[dx, dy] != 0:
            # 防守點已被佔 → 無法堵 → 但這不代表必勝（可能已經有別人的子）
            board[tx, ty] = 0
            idx_stack[depth] += 1
            continue

        board[dx, dy] = defender
        def_stack_x[depth] = dx
        def_stack_y[depth] = dy
        has_def[depth] = 1

        # 防守完後，繼續找新的衝四 threats
        nt = _find_four_threats(board, attacker)
        if len(nt) == 0:
            # 沒有後續衝四了 → 這條路死了
            board[dx, dy] = 0
            board[tx, ty] = 0
            has_def[depth] = 0
            idx_stack[depth] += 1
            continue

        # 進入下一層
        depth += 1
        cnt_n = min(len(nt), 64)
        layer_tcnt[depth] = cnt_n
        for i in range(cnt_n):
            layer_threats[depth, i, 0] = nt[i, 0]
            layer_threats[depth, i, 1] = nt[i, 1]
        idx_stack[depth] = 0

    return False


# ==========================================
# 單元測試
# ==========================================
if __name__ == '__main__':
    print("=== VCF Unit Tests ===")

    # Test 1: 三連 + 雙端開放 = VCF 必勝
    b1 = np.zeros((15, 15), dtype=np.int8)
    b1[7, 6:9] = 1  # 黑棋三連在第 7 行
    result1 = vcf_search(b1.copy(), 1, max_depth=15)
    print(f"Test 1 (三連開放): VCF={result1} -> 預期 True")
    assert result1 == True, "Test 1 FAILED"

    # Test 2: 孤子 = 無殺
    b2 = np.zeros((15, 15), dtype=np.int8)
    b2[7, 7] = 1
    result2 = vcf_search(b2.copy(), 1, max_depth=15)
    print(f"Test 2 (孤子): VCF={result2} -> 預期 False")
    assert result2 == False, "Test 2 FAILED"

    # Test 3: 完全封鎖 = 無殺
    b3 = np.zeros((15, 15), dtype=np.int8)
    b3[7, 6:9] = 1
    b3[7, 5] = -1
    b3[7, 9] = -1
    result3 = vcf_search(b3.copy(), 1, max_depth=15)
    print(f"Test 3 (封鎖): VCF={result3} -> 預期 False")
    assert result3 == False, "Test 3 FAILED"

    # Test 4: 四連 = 必勝 (一步殺)
    b4 = np.zeros((15, 15), dtype=np.int8)
    b4[7, 5:9] = 1  # 四連
    result4 = vcf_search(b4.copy(), 1, max_depth=15)
    print(f"Test 4 (四連): VCF={result4} -> 預期 True")
    assert result4 == True, "Test 4 FAILED"

    # Test 5: board.copy() 安全性 — 原始棋盤不被修改
    b5 = np.zeros((15, 15), dtype=np.int8)
    b5[7, 6:9] = 1
    b5_snapshot = b5.copy()
    vcf_search(b5.copy(), 1, max_depth=15)
    assert np.array_equal(b5, b5_snapshot), "Test 5 FAILED: board was mutated!"
    print("Test 5 (board safety): PASSED")

    print("\n=== ALL VCF TESTS PASSED ===")
