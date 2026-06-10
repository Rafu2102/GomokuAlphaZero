import os
import sys
import time
import math
import threading
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem,
    QGraphicsSimpleTextItem,
    QPushButton, QLabel, QComboBox, QFrame, QTextEdit, QGridLayout, QGraphicsDropShadowEffect,
    QStackedWidget, QSpacerItem, QSizePolicy, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QTimer
from PyQt6.QtGui import QColor, QPen, QBrush, QFont, QPainter, QRadialGradient, QImage, QPixmap, QCursor

import pyqtgraph as pg

from env import GomokuEnv
from mcts import MCTSEngine
from resnet import PolicyValueNet
from prediction_server import BOARD_SIZE

# ==========================================
# 🍵 專業暗色禪意色盤 (Dark Zen Theme)
# ==========================================
APP_BG = "#121212"
PANEL_BG = "#1E1E1E"
PANEL_BORDER = "#333333"
TEXT_MAIN = "#E0E0E0"
TEXT_MUTED = "#888888"
TEXT_ACCENT = "#DAB07F"
WIN_RATE_LINE = "#E53935"
WIN_RATE_FILL = QColor(229, 57, 53, 50) 
WOOD_BASE = QColor("#C78C4C")
GRID_COLOR = QColor("#3D1E04")

CELL_SIZE = 42
BOARD_PX = BOARD_SIZE * CELL_SIZE
MARGIN = 40

os.environ['QT_LOGGING_RULES'] = 'qt.gui.scenegraph=false'

# ==========================================
# 通用函式
# ==========================================
def to_coord(action):
    r = action // BOARD_SIZE
    c = action % BOARD_SIZE
    return f"{chr(ord('A') + c)}{15 - r}"

def create_wood_texture(width, height):
    img = QImage(int(width), int(height), QImage.Format.Format_ARGB32)
    img.fill(WOOD_BASE)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    np.random.seed(42)
    for _ in range(1200):
        alpha = np.random.randint(15, 35)
        thick = np.random.randint(1, 4)
        pen = QPen(QColor(80, 40, 15, alpha), thick)
        painter.setPen(pen)
        
        x = np.random.randint(0, int(width))
        y1 = np.random.randint(-50, int(height))
        y2 = y1 + np.random.randint(50, 400)
        offset_x = np.random.randint(-20, 20)
        painter.drawLine(x, y1, x + offset_x, y2)
        
    painter.end()
    return img

def create_button(text, is_primary=False):
    btn = QPushButton(text)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if is_primary:
        btn.setStyleSheet(f"""
            QPushButton {{ background-color: {TEXT_ACCENT}; color: #121212; border-radius: 8px; padding: 14px; font-weight: bold; font-family: 'Microsoft JhengHei'; font-size: 16px; letter-spacing: 1px;}}
            QPushButton:hover {{ background-color: #E2B98A; }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{ background-color: transparent; color: {TEXT_MAIN}; border-radius: 8px; border: 1px solid #555; padding: 14px; font-weight: bold; font-family: 'Microsoft JhengHei'; font-size: 15px; letter-spacing: 1px;}}
            QPushButton:hover {{ background-color: rgba(255, 255, 255, 0.05); }}
        """)
    return btn

def prob_to_color(p):
    r = int(255 * p)
    b = int(255 * (1 - p))
    g = int(80 * (1 - abs(p - 0.5)*2))
    return QColor(r, g, b, int(220 * p))

# ==========================================
# 🧵 非同步 AI 執行緒
# ==========================================
# 難度設定統一常數表 (所有地方共用，避免不一致)
DIFF_PLAYOUT_MAP = {
    "休閒 (Easy)": 20,
    "挑戰 (Medium)": 150,
    "深淵 (Hard)": 400,
    "極限 (Extreme)": 1000,
}
DIFF_TEMP_MAP = {
    "休閒 (Easy)": 1.0,
    "挑戰 (Medium)": 0.1,
    "深淵 (Hard)": 1e-3,
    "極限 (Extreme)": 1e-3,
}

class AIEngineWorker(QThread):
    finished_signal = pyqtSignal(int, float, list, list, list)
    
    def __init__(self, env, model, device, mcts, difficulty_level, model_lock, worker_id=0):
        super().__init__()
        self.env = env.clone()
        self.model = model
        self.device = device
        self.worker_id = worker_id
        self.model_lock = model_lock
        # 🔒 共享 MCTS 實例（保留 Subtree Retention 子樹繼承，這是 AI 棋力的核心）
        self.mcts = mcts
        self.mcts.n_playout = DIFF_PLAYOUT_MAP.get(difficulty_level, 150)
        
        self.temp = DIFF_TEMP_MAP.get(difficulty_level, 0.1)
        
        # 🚀 GPU Tensor Buffer: 預分配，避免每次 inference 都 new + copy
        self._input_buffer = torch.zeros(1, 4, BOARD_SIZE, BOARD_SIZE, device=device)

    def predict_fn(self, state_tensor):
        # 將 numpy 複製到預先分配的 GPU buffer 中
        self._input_buffer[0].copy_(torch.from_numpy(state_tensor))
        
        with self.model_lock:
            with torch.no_grad():
                log_probs, value = self.model(self._input_buffer)
                probs = torch.exp(log_probs).cpu().numpy()[0]
                v = float(value.cpu().numpy()[0][0])
        return list(enumerate(probs)), v

    def run(self):
        # 🎯 AI 先手第一步：直接下天元 (H8)，省略 MCTS 搜尋
        if np.count_nonzero(self.env.board) == 0:
            center = BOARD_SIZE // 2
            action = center * BOARD_SIZE + center
            self.finished_signal.emit(action, 0.5, ["H8"], [], [])
            return
        
        # 🎯 探索噪聲控制：保持最強棋力，不加噪聲
        acts, probs = self.mcts.get_action_probs(self.env, self.predict_fn, temperature=self.temp, dirichlet_alpha=0)
        
        # 🛡️ 安全防禦：如果 MCTS 未展開或回傳空列表，退化至環境合法步中隨機落子，防止崩潰
        if len(acts) == 0:
            legal_moves = self.env.get_legal_moves()
            if len(legal_moves) > 0:
                action = int(np.random.choice(legal_moves))
            else:
                action = 0
        elif np.sum(probs) == 0:
            action = int(np.random.choice(acts))
        else:
            action = int(acts[np.argmax(probs)])
        
        # 取得 Principal Variation (PV)
        pv_path = [to_coord(a) for a in self.mcts.cpp_tree.get_pv_path()]
            
        # q_value 始終代表等待被下子的「對手」勝率
        win_rate_for_opponent = (self.mcts.cpp_tree.get_root_q_value() + 1.0) / 2.0 
        
        if self.env.current_player == 1:
            # 輪到黑子思考 -> 對手是白子，所以是對手(白子)的勝率
            black_win_rate = 1.0 - win_rate_for_opponent
        else:
            # 輪到白子思考 -> 對手是黑子，所以是對手(黑子)的勝率
            black_win_rate = win_rate_for_opponent
        
        self.finished_signal.emit(action, float(black_win_rate), pv_path, list(acts), list(probs))

class HeatmapWorker(QThread):
    finished_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, env, model, device, model_lock):
        super().__init__()
        self.env = env.clone()
        self.model = model
        self.device = device
        self.model_lock = model_lock
        self._input_buffer = torch.zeros(1, 4, BOARD_SIZE, BOARD_SIZE, device=device)
        
    def run(self):
        with self.model_lock:
            with torch.no_grad():
                if self.device.type == "cuda":
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    from contextlib import nullcontext
                    autocast_ctx = nullcontext()
                
                with autocast_ctx:
                    state_tensor = self.env._get_obs()
                    self._input_buffer[0].copy_(torch.from_numpy(state_tensor))
                    log_probs, _ = self.model(self._input_buffer)
                probs = torch.exp(log_probs).cpu().numpy()[0]
        self.finished_signal.emit(probs)

# ==========================================
# 🌿 3D 立體雲子
# ==========================================
class YunziPiece(QGraphicsEllipseItem):
    def __init__(self, r, c, is_black, is_new=True, is_ghost=False, is_forbidden=False):
        super().__init__(-CELL_SIZE//2 + 1, -CELL_SIZE//2 + 1, CELL_SIZE - 2, CELL_SIZE - 2)
        self.setPos(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE)
        self.setTransformOriginPoint(0, 0)
        
        if is_forbidden:
            self.setPen(QPen(QColor("#FF3B30"), 1.8, Qt.PenStyle.DashLine))
            self.setBrush(QBrush(QColor(255, 59, 48, 30)))
            self.x_line1 = QGraphicsLineItem(-CELL_SIZE//4, -CELL_SIZE//4, CELL_SIZE//4, CELL_SIZE//4, self)
            self.x_line2 = QGraphicsLineItem(CELL_SIZE//4, -CELL_SIZE//4, -CELL_SIZE//4, CELL_SIZE//4, self)
            pen = QPen(QColor("#FF3B30"), 2.5)
            self.x_line1.setPen(pen)
            self.x_line2.setPen(pen)
            self.setZValue(3)
            return
            
        self.setPen(QPen(Qt.PenStyle.NoPen))
        
        hl_x, hl_y = -CELL_SIZE//4, -CELL_SIZE//4
        gradient = QRadialGradient(hl_x, hl_y, CELL_SIZE//1.3, hl_x, hl_y)
        if is_black:
            gradient.setColorAt(0.0, QColor("#555555"))
            gradient.setColorAt(0.3, QColor("#1A1A1A"))
            gradient.setColorAt(1.0, QColor("#000000"))
        else:
            gradient.setColorAt(0.0, QColor("#FFFFFF"))
            gradient.setColorAt(0.5, QColor("#EAEAEA"))
            gradient.setColorAt(1.0, QColor("#A0A0A0"))
            
        self.setBrush(QBrush(gradient))
        
        if is_ghost:
            self.setOpacity(0.4)
            self.setZValue(3)
        else:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)
            shadow.setOffset(2, 4)
            shadow.setColor(QColor(0, 0, 0, 160))
            self.setGraphicsEffect(shadow)
            
            if is_new:
                self.setZValue(2)
                self.anim_step = 0
                self.anim_max = 8 
                self.setScale(1.2)
                self.setOpacity(0.5)
                self.timer = QTimer()
                self.timer.timeout.connect(self._animate_drop)
                self.timer.start(15)
            else:
                self.setZValue(1)

    def _animate_drop(self):
        self.anim_step += 1
        p = self.anim_step / self.anim_max
        self.setScale(1.2 - (0.2 * p))
        self.setOpacity(0.5 + 0.5 * p)
        if self.anim_step >= self.anim_max:
            self.setScale(1.0)
            self.setOpacity(1.0)
            self.timer.stop()

class LastIndicatorItem(QGraphicsEllipseItem):
    def __init__(self, cx, cy):
        super().__init__(-15, -15, 30, 30)
        self.setPos(cx, cy)
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(4)
        
        self.angle = 0.0
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate_breath)
        self.timer.start(30)
        
    def _animate_breath(self):
        self.angle += 0.1
        self.update()
        
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = 5.0 + 5.0 * (1.0 + math.sin(self.angle)) / 2.0
        
        gradient = QRadialGradient(0, 0, radius + 4)
        gradient.setColorAt(0.0, QColor(255, 51, 51, 230))
        gradient.setColorAt(0.4, QColor(255, 51, 51, 100))
        gradient.setColorAt(1.0, QColor(255, 51, 51, 0))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QPointF(0, 0), radius + 4, radius + 4)
        
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPointF(0, 0), 2.5, 2.5)

class BoardView(QGraphicsView):
    def __init__(self, scene, parent_gui):
        super().__init__(scene)
        self.parent_gui = parent_gui
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setStyleSheet("border: none; background: transparent;")
        
    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        self.parent_gui._handle_mouse_move(pos)
        super().mouseMoveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        self.parent_gui._handle_mouse_click(pos)
        super().mousePressEvent(event)
        
    def leaveEvent(self, event):
        self.parent_gui._clear_ghost()
        super().leaveEvent(event)


# ==========================================
# 🎮 頁面1：主選單 (Main Menu)
# ==========================================
class MainMenu(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setStyleSheet(f"background-color: {APP_BG};")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("ALPHAZERO\nGOMOKU")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_MAIN}; font-family: 'Consolas'; font-size: 80px; font-weight: 900; letter-spacing: 12px;")
        layout.addWidget(title)
        
        subtitle = QLabel("神經網路五子棋對弈終端")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; font-family: 'Microsoft JhengHei'; font-size: 26px; font-weight: bold; letter-spacing: 15px;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(45)
        
        control_panel = QFrame()
        control_panel.setFixedWidth(640)
        control_panel.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 20px; border: 1px solid {PANEL_BORDER};")
        cp_layout = QVBoxLayout(control_panel)
        cp_layout.setContentsMargins(50, 50, 50, 50)
        cp_layout.setSpacing(30)
        
        diff_lbl = QLabel("▍ AI 算力配置 (DIFFICULTY)")
        diff_lbl.setStyleSheet(f"color: {TEXT_ACCENT}; font-size: 14px; font-weight: 900; letter-spacing: 3px;")
        cp_layout.addWidget(diff_lbl)
        
        self.diff_buttons = []
        self.current_diff = "挑戰 (Medium)"
        
        diff_box = QHBoxLayout()
        diff_box.setSpacing(15)
        
        diffs = [
            ("休閒", "Easy\n(20 Playouts)", "休閒 (Easy)"),
            ("挑戰", "Medium\n(150 Playouts)", "挑戰 (Medium)"),
            ("深淵", "Hard\n(400 Playouts)", "深淵 (Hard)"),
            ("極限", "Extreme\n(1000 Playouts)", "極限 (Extreme)")
        ]
        
        for name, sub, actual_diff in diffs:
            btn = QPushButton(f"{name}\n{sub}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("diff_val", actual_diff)
            btn.clicked.connect(self._on_diff_clicked)
            self.diff_buttons.append(btn)
            diff_box.addWidget(btn)
            
        cp_layout.addLayout(diff_box)
        self._update_diff_styles()
        
        cp_layout.addSpacing(10)
        
        btn_black = create_button("🧑‍💻 啟動對弈 ➔ 人類戰 AI (執黑)", is_primary=True)
        btn_black.clicked.connect(lambda: self.start("PvAI", 1))
        
        btn_white = create_button("🛡️ 啟動對弈 ➔ 人類戰 AI (執白)", is_primary=False)
        btn_white.clicked.connect(lambda: self.start("PvAI", -1))
        
        btn_pvp = create_button("🙋 啟動對弈 ➔ 兩人遊玩 (PvP)", is_primary=False)
        btn_pvp.clicked.connect(lambda: self.start("PvP", 1))
        
        btn_aivai = create_button("🤖 啟動對弈 ➔ AI 互打觀戰", is_primary=False)
        btn_aivai.clicked.connect(lambda: self.start("AIvAI", 1))
        
        cp_layout.addWidget(btn_black)
        cp_layout.addWidget(btn_white)
        cp_layout.addWidget(btn_pvp)
        cp_layout.addWidget(btn_aivai)
        
        layout.addWidget(control_panel, alignment=Qt.AlignmentFlag.AlignHCenter)

    def _on_diff_clicked(self):
        sender = self.sender()
        self.current_diff = sender.property("diff_val")
        self._update_diff_styles()

    def _update_diff_styles(self):
        for btn in self.diff_buttons:
            if btn.property("diff_val") == self.current_diff:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {TEXT_ACCENT};
                        color: #121212;
                        border-radius: 8px;
                        padding: 15px;
                        font-weight: 900;
                        font-family: 'Microsoft JhengHei', 'Segoe UI';
                        font-size: 14px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgba(255, 255, 255, 0.05);
                        color: {TEXT_MUTED};
                        border-radius: 8px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        padding: 15px;
                        font-weight: bold;
                        font-family: 'Microsoft JhengHei', 'Segoe UI';
                        font-size: 13px;
                    }}
                    QPushButton:hover {{
                        background-color: rgba(255, 255, 255, 0.1);
                        color: {TEXT_MAIN};
                    }}
                """)

    def start(self, game_mode, human_color):
        self.main_window.start_game(game_mode, human_color, self.current_diff)

# ==========================================
# 🎮 頁面2：對弈大廳 (Game Board)
# ==========================================
class GameBoard(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setStyleSheet(f"background-color: {APP_BG};")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.env = GomokuEnv()
        self.device = self.main_window.device
        self.model = self.main_window.model
        
        self.game_mode = "PvAI"
        self.human_player = 1
        self.ai_thinking = False
        self.ghost_piece = None
        self.current_diff = "挑戰 (Medium)"
        self.win_rate_history = []
        self.action_history = []
        self.piece_items = []
        self.heatmap_items = []
        self.last_ai_mcts_probs = None
        self.forbidden_tooltip = None
        
        self.mcts = None
        self.last_indicator = None
        self.win_line_items = []
        
        self.thinking_dots = 0
        self.think_timer = QTimer(self)
        self.think_timer.timeout.connect(self._animate_thinking)
        
        # 🔥 熱力圖即時刷新器 (按住 Shift 時每 200ms 更新一次)
        self._heatmap_live = False
        self._heatmap_timer = QTimer(self)
        self._heatmap_timer.timeout.connect(self._refresh_heatmap)
        
        self._init_ui()

    def _animate_thinking(self):
        self.thinking_dots = (self.thinking_dots + 1) % 4
        self.status_lbl.setText("🧠 深度神經解算中" + "." * self.thinking_dots)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        board_container = QFrame()
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 180))
        board_container.setGraphicsEffect(shadow)
        
        board_layout = QVBoxLayout(board_container)
        board_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scene = QGraphicsScene()
        board_size_px = BOARD_PX + MARGIN * 2
        self.scene.setSceneRect(0, 0, board_size_px, board_size_px)
        self.view = BoardView(self.scene, self)
        self.view.setMinimumSize(400, 400)
        board_layout.addWidget(self.view)
        
        wood_img = create_wood_texture(BOARD_PX + MARGIN*2, BOARD_PX + MARGIN*2)
        self.bg_item = QGraphicsRectItem(0, 0, BOARD_PX + MARGIN*2, BOARD_PX + MARGIN*2)
        self.bg_item.setBrush(QBrush(wood_img))
        self.scene.addItem(self.bg_item)
        
        pen = QPen(GRID_COLOR, 1.5)
        for i in range(BOARD_SIZE):
            self.scene.addLine(MARGIN, MARGIN + i * CELL_SIZE, MARGIN + BOARD_PX - CELL_SIZE, MARGIN + i * CELL_SIZE, pen)
            self.scene.addLine(MARGIN + i * CELL_SIZE, MARGIN, MARGIN + i * CELL_SIZE, MARGIN + BOARD_PX - CELL_SIZE, pen)
        
        stars = [(3,3), (11,3), (3,11), (11,11), (7,7)]
        for r, c in stars:
            cx = MARGIN + c * CELL_SIZE
            cy = MARGIN + r * CELL_SIZE
            self.scene.addEllipse(cx - 3, cy - 3, 6, 6, QPen(Qt.PenStyle.NoPen), QBrush(GRID_COLOR))
            
        main_layout.addWidget(board_container)
        
        dashboard = QFrame()
        dashboard.setFixedWidth(440)
        dashboard.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 20px; border: 1px solid {PANEL_BORDER};")
        
        shadow_dash = QGraphicsDropShadowEffect()
        shadow_dash.setBlurRadius(30)
        shadow_dash.setColor(QColor(0,0,0, 180))
        shadow_dash.setOffset(0, 5)
        dashboard.setGraphicsEffect(shadow_dash)
        
        dash_layout = QVBoxLayout(dashboard)
        dash_layout.setContentsMargins(30, 30, 30, 30)
        dash_layout.setSpacing(20)
        
        # --- HEADER ---
        btn_back = QPushButton("⏴ 退出作戰區")
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.setStyleSheet(f"""
            QPushButton {{ color: {TEXT_MUTED}; background: transparent; border: none; text-align: left; font-size: 15px; font-weight: 900; letter-spacing: 2px;}}
            QPushButton:hover {{ color: {TEXT_ACCENT}; }}
        """)
        btn_back.clicked.connect(self.main_window.return_to_menu)
        dash_layout.addWidget(btn_back)
        
        status_box = QFrame()
        status_box.setStyleSheet(f"background-color: rgba(218, 176, 127, 0.05); border-radius: 12px; border: 1px solid rgba(218, 176, 127, 0.2);")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(15, 20, 15, 20)
        
        self.status_lbl = QLabel("SYSTEM IDLE")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.setStyleSheet(f"""
            color: #FFFFFF; font-size: 28px; font-weight: 900; font-family: 'Segoe UI', Arial, sans-serif;
            background: transparent; border: none;
        """)
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(20)
        glow.setColor(QColor(TEXT_ACCENT))
        glow.setOffset(0,0)
        self.status_lbl.setGraphicsEffect(glow)
        status_layout.addWidget(self.status_lbl)
        dash_layout.addWidget(status_box)
        
        # --- TACTICAL ADVANTAGE CARD ---
        wr_card = QFrame()
        wr_card.setStyleSheet(f"background-color: rgba(10, 10, 10, 0.8); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);")
        wr_layout = QVBoxLayout(wr_card)
        wr_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_wr = QLabel("▍ TACTICAL ADVANTAGE (勝率預測)")
        lbl_wr.setStyleSheet(f"color: {TEXT_ACCENT}; font-size: 12px; font-weight: 900; letter-spacing: 2px; background: transparent; border: none;")
        wr_layout.addWidget(lbl_wr)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('transparent')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color=TEXT_MUTED))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color=TEXT_MAIN))
        self.plot_widget.setYRange(0, 1)
        self.plot_widget.setFixedHeight(120)
        self.plot_widget.setStyleSheet("border: none; background: transparent;")
        
        pen = pg.mkPen(color=WIN_RATE_LINE, width=4)
        brush = pg.mkBrush(WIN_RATE_FILL)
        self.plot_curve = self.plot_widget.plot([], [], pen=pen, fillLevel=0, brush=brush)
        wr_layout.addWidget(self.plot_widget)
        dash_layout.addWidget(wr_card)
        
        # --- CHRONO SHIFT CARD ---
        tt_card = QFrame()
        tt_card.setStyleSheet(f"background-color: rgba(10, 10, 10, 0.8); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);")
        tt_layout = QVBoxLayout(tt_card)
        tt_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_tt = QLabel("▍ CHRONO SHIFT (時空裂隙)")
        lbl_tt.setStyleSheet(f"color: {TEXT_ACCENT}; font-size: 12px; font-weight: 900; letter-spacing: 2px; background: transparent; border: none;")
        tt_layout.addWidget(lbl_tt)
        
        slider_box = QHBoxLayout()
        slider_box.setSpacing(15)
        self.slider_time = QSlider(Qt.Orientation.Horizontal)
        self.slider_time.setMinimum(0)
        self.slider_time.setMaximum(0)
        self.slider_time.setValue(0)
        self.slider_time.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_time.setTickInterval(1)
        self.slider_time.setSingleStep(1)
        self.slider_time.setPageStep(1)
        self.slider_time.setStyleSheet("""
            QSlider { background: transparent; border: none; }
            QSlider::groove:horizontal { border-radius: 4px; height: 8px; background: #333; }
            QSlider::handle:horizontal { background: #DAB07F; width: 18px; margin: -5px 0; border-radius: 9px; }
        """)
        self.slider_time.valueChanged.connect(self._on_slider_changed)
        slider_box.addWidget(self.slider_time)
        
        self.btn_rewind = QPushButton("覆寫歷史")
        self.btn_rewind.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rewind.setStyleSheet(f"""
            QPushButton {{ background-color: #2D2D2D; color: {TEXT_MAIN}; border-radius: 6px; border: 1px solid #666; padding: 10px 15px; font-weight: bold; font-size: 13px; letter-spacing: 1px;}}
            QPushButton:hover {{ background-color: {TEXT_ACCENT}; color: #000; border: 1px solid {TEXT_ACCENT};}}
        """)
        self.btn_rewind.clicked.connect(self._do_time_travel)
        slider_box.addWidget(self.btn_rewind)
        tt_layout.addLayout(slider_box)
        dash_layout.addWidget(tt_card)
        
        # --- NEURAL PRECOGNITION CARD ---
        pv_card = QFrame()
        pv_card.setStyleSheet(f"background-color: rgba(10, 10, 10, 0.8); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);")
        pv_layout = QVBoxLayout(pv_card)
        pv_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_pv = QLabel("▍ NEURAL PRECOGNITION (神經推演)")
        lbl_pv.setStyleSheet(f"color: {TEXT_ACCENT}; font-size: 12px; font-weight: 900; letter-spacing: 2px; background: transparent; border: none;")
        pv_layout.addWidget(lbl_pv)
        
        self.pv_log = QTextEdit()
        self.pv_log.setReadOnly(True)
        self.pv_log.setStyleSheet(f"background-color: #080808; color: #00FF66; border: 1px solid #222; border-radius: 8px; padding: 12px; font-family: Consolas; font-size: 15px; line-height: 1.6;")
        pv_layout.addWidget(self.pv_log)
        
        hint_lbl = QLabel("SHIFT 鍵可啟動上帝之眼熱力圖")
        hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_lbl.setStyleSheet(f"color: #666666; font-size: 11px; font-weight: bold; background: transparent; border: none; margin-top: 5px;")
        pv_layout.addWidget(hint_lbl)
        
        dash_layout.addWidget(pv_card)
        main_layout.addWidget(dashboard)

    # ================= MCTS 上帝之眼 =================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            self._heatmap_live = True
            self._show_heatmap()
            self._heatmap_timer.start(200)  # 每 200ms 刷新一次
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            self._heatmap_live = False
            self._heatmap_timer.stop()
            self._hide_heatmap()
        super().keyReleaseEvent(event)
    
    def _refresh_heatmap(self):
        """Timer 回呼：持續刷新熱力圖（只在按住 Shift 時活動）"""
        if self._heatmap_live:
            if self.last_ai_mcts_probs is None:
                self._show_heatmap()
        
    def _show_heatmap(self):
        if self.env.done or self.ai_thinking:
            return
            
        if self.last_ai_mcts_probs is not None:
            self._draw_heatmap_data(self.last_ai_mcts_probs, is_mcts=True)
        else:
            if getattr(self, 'heatmap_worker', None) is not None:
                try:
                    if self.heatmap_worker.isRunning():
                        return
                except RuntimeError:
                    self.heatmap_worker = None
            
            self.heatmap_worker = HeatmapWorker(self.env, self.model, self.device, self.main_window.model_lock)
            self.heatmap_worker.finished_signal.connect(self._on_heatmap_finished)
            self.heatmap_worker.finished_signal.connect(self.heatmap_worker.deleteLater)
            self.heatmap_worker.start()

    def _on_heatmap_finished(self, probs):
        if not self._heatmap_live or self.env.done or self.ai_thinking:
            return
        self._draw_heatmap_data(probs, is_mcts=False)

    def _draw_heatmap_data(self, probs, is_mcts=False):
        self._hide_heatmap()
        if self.env.done or self.ai_thinking:
            return
            
        legal_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
        legal_moves = self.env.get_legal_moves()
        legal_probs[legal_moves] = probs[legal_moves]
        
        raw_probs = legal_probs.copy()
        
        if np.sum(legal_probs) > 0:
            legal_probs /= np.max(legal_probs) 
            
        top_k_indices = np.argsort(legal_probs)[-3:]
        
        for act in legal_moves:
            prob = legal_probs[act]
            if prob < 0.03: continue
            r = act // BOARD_SIZE
            c = act % BOARD_SIZE
            
            rect = QGraphicsRectItem(MARGIN + c*CELL_SIZE - CELL_SIZE//2 + 2, 
                                     MARGIN + r*CELL_SIZE - CELL_SIZE//2 + 2, 
                                     CELL_SIZE - 4, CELL_SIZE - 4)
            rect.setBrush(QBrush(prob_to_color(prob)))
            rect.setPen(QPen(Qt.PenStyle.NoPen))
            rect.setZValue(5)
            self.scene.addItem(rect)
            self.heatmap_items.append(rect)
            
            if act in top_k_indices:
                circle = QGraphicsEllipseItem(
                    MARGIN + c*CELL_SIZE - 20,
                    MARGIN + r*CELL_SIZE - 20,
                    40, 40
                )
                circle.setPen(QPen(QColor("#FFD700"), 3))
                circle.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                circle.setZValue(6)
                self.scene.addItem(circle)
                self.heatmap_items.append(circle)
                
                if is_mcts:
                    label_text = f"{int(raw_probs[act] * self.mcts.n_playout)}"
                else:
                    label_text = f"{raw_probs[act]*100:.0f}%"
                
                text_item = QGraphicsSimpleTextItem(label_text)
                text_item.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
                text_item.setBrush(QBrush(QColor("#FFFFFF")))
                text_item.setZValue(7)
                text_rect = text_item.boundingRect()
                text_item.setPos(
                    MARGIN + c*CELL_SIZE - text_rect.width()/2,
                    MARGIN + r*CELL_SIZE - text_rect.height()/2
                )
                self.scene.addItem(text_item)
                self.heatmap_items.append(text_item)
            
    def _hide_heatmap(self):
        for item in getattr(self, 'heatmap_items', []):
            try:
                self.scene.removeItem(item)
            except Exception:
                pass
        self.heatmap_items = []

    def check_forbidden_type(self, r, c):
        from board_utils import _get_ray, _analyze_ray, _verify_suspect_three
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        total_fours = 0
        total_suspect_threes = 0
        three_dir_flags = np.zeros(4, dtype=np.int8)
        
        for d_idx in range(4):
            dx, dy = directions[d_idx]
            ray = _get_ray(self.env.board, r, c, dx, dy)
            is_five, is_overline, s_threes, fours = _analyze_ray(ray, 1)
            
            if is_overline:
                return "長連禁手"
                
            total_fours += fours
            total_suspect_threes += s_threes
            if s_threes > 0:
                three_dir_flags[d_idx] = 1
                
        if total_fours >= 2:
            return "四四禁手"
            
        if total_suspect_threes >= 2:
            verified_threes = 0
            for d_idx in range(4):
                if three_dir_flags[d_idx] == 1:
                    dx, dy = directions[d_idx]
                    if _verify_suspect_three(self.env.board, r, c, dx, dy, 1):
                        verified_threes += 1
            if verified_threes >= 2:
                return "三三禁手"
                
        return "禁手"

    # ================= 時光機與生命週期 =================
    def setup_game(self, game_mode, human_color, diff):
        self.game_mode = game_mode
        self.human_player = human_color
        self.current_diff = diff
        self.env.reset()
        
        self.win_rate_history = [0.5]
        self.action_history = []
        self.piece_items = []
        self._hide_heatmap()
        self._clear_win_lines()
        
        self.pv_log.clear()
        self.plot_curve.setData([])
        self._clear_ghost()
        self.ai_thinking = False
        self._worker_gen = getattr(self, '_worker_gen', 0) + 1  # 🔒 世代識別碼
        
        n_playout = DIFF_PLAYOUT_MAP.get(diff, 150)
        self.mcts = MCTSEngine(c_puct=2.0, n_playout=n_playout)
        
        for item in self.scene.items():
            if isinstance(item, YunziPiece) or isinstance(item, LastIndicatorItem) or item is self.last_indicator:
                if hasattr(item, 'timer') and item.timer:
                    item.timer.stop()
                try:
                    self.scene.removeItem(item)
                except Exception:
                    pass
        self.last_indicator = None

        self.slider_time.setMaximum(0)
        self.slider_time.setValue(0)
        self.slider_time.setEnabled(True)

        if self._is_human_turn():
            self.status_lbl.setText("等待玩家落子")
        else:
            self.status_lbl.setText("AI 準備中...")
            
        self.setFocus()
        
        if self._is_ai_turn():
            self._trigger_ai_turn()

    def _is_human_turn(self):
        if self.game_mode == "PvP":
            return True
        if self.game_mode == "PvAI" and self.env.current_player == self.human_player:
            return True
        return False

    def _is_ai_turn(self):
        if self.game_mode == "AIvAI":
            return True
        if self.game_mode == "PvAI" and self.env.current_player != self.human_player:
            return True
        return False

    def _on_slider_changed(self, value):
        for i, piece in enumerate(self.piece_items):
            if i < value:
                piece.setOpacity(1.0)
            else:
                piece.setOpacity(0.15) 
                
        if self.last_indicator:
            self.last_indicator.setVisible(value == len(self.action_history))

    def _do_time_travel(self):
        # 💡 新增防禦：禁止在 AI 運算時擾動時空
        if self.ai_thinking:
            self.pv_log.append("⚠️ 警告：AI 正在觀測未來，無法擾動時空！\n")
            self.pv_log.verticalScrollBar().setValue(self.pv_log.verticalScrollBar().maximum())
            return
            
        target_turn = self.slider_time.value()
        if target_turn == len(self.action_history):
            self.pv_log.append("⚠️ 即是最新的現實，無歷史可覆寫。\n")
            return
            
        self.main_window.shake(intensity=12, duration=400)
        
        self.action_history = self.action_history[:target_turn]
        self.win_rate_history = self.win_rate_history[:target_turn+1]
        self.plot_curve.setData(self.win_rate_history)
        
        self.env.reset()
        n_playout = DIFF_PLAYOUT_MAP.get(self.current_diff, 150)
        self.mcts = MCTSEngine(c_puct=2.0, n_playout=n_playout)
        
        self.last_ai_mcts_probs = None
        for item in self.scene.items():
            if isinstance(item, YunziPiece) or isinstance(item, LastIndicatorItem) or item is self.last_indicator:
                if hasattr(item, 'timer') and item.timer:
                    item.timer.stop()
                try:
                    self.scene.removeItem(item)
                except Exception:
                    pass
        self.last_indicator = None
        self.piece_items = []
        self._clear_win_lines()
        
        self._clear_ghost()
        self.pv_log.append("\n⏳ [時光機] 歷史線已切斷，重建殘局...\n")
        
        for action in self.action_history:
            self.env.step(action)
            self.mcts.update_with_move(action)
            is_black = (self.env.board[action // BOARD_SIZE, action % BOARD_SIZE] == 1)
            self._sync_board_ui(new_action=action, is_black=is_black, is_fast_forward=True)
            
        self.slider_time.setMaximum(len(self.action_history))
        
        self.ai_thinking = False
        self._check_game_state()
        if not self.env.done:
            # Check whose turn it is in the new timeline!
            if self._is_ai_turn():
                self._trigger_ai_turn()
            else:
                self.status_lbl.setText("等待玩家落子")
                self.pv_log.append("👉 現在輪到您下棋了\n")

    # ================= 五子連線動畫 =================
    def _clear_win_lines(self):
        for item in getattr(self, 'win_line_items', []):
            self.scene.removeItem(item)
        self.win_line_items = []

    def _find_winning_line(self):
        winner = self.env.winner
        if winner == 0: return []
        b = self.env.board
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if b[r,c] == winner:
                    for dr, dc in dirs:
                        line = []
                        for i in range(5):
                            nr, nc = r + dr*i, c + dc*i
                            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and b[nr, nc] == winner:
                                line.append((nr, nc))
                            else:
                                break
                        if len(line) == 5:
                            return line
        return []

    def _play_win_animation(self, win_line):
        r1, c1 = win_line[0]
        r2, c2 = win_line[-1]
        
        line = QGraphicsLineItem(MARGIN + c1*CELL_SIZE, MARGIN + r1*CELL_SIZE, 
                                 MARGIN + c1*CELL_SIZE, MARGIN + r1*CELL_SIZE)
        
        color = QColor(255, 50, 50, 220) if self.env.winner == -self.human_player else QColor(255, 215, 0, 220)
        pen = QPen(color)
        pen.setWidth(8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        line.setPen(pen)
        line.setZValue(6) 
        
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(20)
        glow.setColor(color)
        glow.setOffset(0,0)
        line.setGraphicsEffect(glow)
        
        self.scene.addItem(line)
        self.win_line_items.append(line)
        
        self.win_anim_step = 0
        self.win_anim_max = 25 
        self.win_line_item = line
        self.win_target_x = MARGIN + c2*CELL_SIZE
        self.win_target_y = MARGIN + r2*CELL_SIZE
        self.win_start_x = MARGIN + c1*CELL_SIZE
        self.win_start_y = MARGIN + r1*CELL_SIZE
        
        self.win_timer = QTimer(self)
        self.win_timer.timeout.connect(self._animate_win_line)
        self.win_timer.start(35) 

    def _animate_win_line(self):
        self.win_anim_step += 1
        p = self.win_anim_step / self.win_anim_max
        # Smooth ease-out
        p = p * (2 - p)
        cx = self.win_start_x + (self.win_target_x - self.win_start_x) * p
        cy = self.win_start_y + (self.win_target_y - self.win_start_y) * p
        self.win_line_item.setLine(self.win_start_x, self.win_start_y, cx, cy)
        
        if self.win_anim_step >= self.win_anim_max:
            self.win_timer.stop()


    # ================= 遊戲核心互動 =================
    def _clear_ghost(self):
        if self.ghost_piece is not None:
            try:
                self.scene.removeItem(self.ghost_piece)
            except Exception:
                pass
            self.ghost_piece = None
        self.ghost_was_forbidden = False
        self._clear_forbidden_tooltip()

    def _show_forbidden_tooltip(self, r, c, ban_type):
        self._clear_forbidden_tooltip()
        
        bg = QGraphicsRectItem()
        bg.setBrush(QBrush(QColor("#1E1E1E")))
        bg.setPen(QPen(QColor("#FF3B30"), 1))
        
        text = QGraphicsSimpleTextItem(f"⚠️ {ban_type}")
        text.setFont(QFont("Microsoft JhengHei", 10, QFont.Weight.Bold))
        text.setBrush(QBrush(QColor("#FF3B30")))
        
        pad = 6
        rect = text.boundingRect()
        bg.setRect(0, 0, rect.width() + pad * 2, rect.height() + pad * 2)
        text.setPos(pad, pad)
        text.setParentItem(bg)
        
        px = MARGIN + c * CELL_SIZE + 15
        py = MARGIN + r * CELL_SIZE - 35
        bg.setPos(px, py)
        bg.setZValue(10)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(1, 2)
        bg.setGraphicsEffect(shadow)
        
        self.scene.addItem(bg)
        self.forbidden_tooltip = bg
        
    def _clear_forbidden_tooltip(self):
        if hasattr(self, 'forbidden_tooltip') and self.forbidden_tooltip:
            try:
                self.scene.removeItem(self.forbidden_tooltip)
            except Exception:
                pass
            self.forbidden_tooltip = None

    def _handle_mouse_move(self, pos):
        if self.ai_thinking or self.env.done or not self._is_human_turn():
            self._clear_ghost()
            return
        
        if self.slider_time.value() != len(self.action_history):
            self._clear_ghost()
            return
            
        c = round((pos.x() - MARGIN) / CELL_SIZE)
        r = round((pos.y() - MARGIN) / CELL_SIZE)
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.env.board[r, c] == 0:
            action = r * BOARD_SIZE + c
            is_forbidden = (self.env.current_player == 1 and action not in self.env.get_legal_moves())
            
            if is_forbidden:
                if self.ghost_piece is None or not getattr(self, 'ghost_was_forbidden', False):
                    if self.ghost_piece is not None:
                        try:
                            self.scene.removeItem(self.ghost_piece)
                        except Exception:
                            pass
                    self.ghost_piece = YunziPiece(r, c, is_black=True, is_ghost=True, is_forbidden=True)
                    self.scene.addItem(self.ghost_piece)
                    self.ghost_was_forbidden = True
                else:
                    self.ghost_piece.setPos(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE)
                
                ban_type = self.check_forbidden_type(r, c)
                self._show_forbidden_tooltip(r, c, ban_type)
            else:
                self._clear_forbidden_tooltip()
                is_black = (self.env.current_player == 1)
                if self.ghost_piece is None or getattr(self, 'ghost_was_forbidden', False):
                    if self.ghost_piece is not None:
                        try:
                            self.scene.removeItem(self.ghost_piece)
                        except Exception:
                            pass
                    self.ghost_piece = YunziPiece(r, c, is_black, is_ghost=True)
                    self.scene.addItem(self.ghost_piece)
                    self.ghost_was_forbidden = False
                else:
                    self.ghost_piece.setPos(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE)
        else:
            self._clear_ghost()

    def _handle_mouse_click(self, pos):
        if self.ai_thinking or self.env.done or not self._is_human_turn():
            return
            
        if self.slider_time.value() != len(self.action_history):
            self.pv_log.append("⚠️ 警告：時間線錯亂，請先點擊【確認回溯】\n")
            self.pv_log.verticalScrollBar().setValue(self.pv_log.verticalScrollBar().maximum())
            return
            
        c = round((pos.x() - MARGIN) / CELL_SIZE)
        r = round((pos.y() - MARGIN) / CELL_SIZE)
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            action = r * BOARD_SIZE + c
            if action in self.env.get_legal_moves():
                self._clear_ghost()
                self.last_ai_mcts_probs = None  # 人類落子，重置前一手 AI 思考軌跡
                
                self.action_history.append(action)
                self.slider_time.setMaximum(len(self.action_history))
                self.slider_time.setValue(len(self.action_history))
                
                self.env.step(action)
                self.mcts.update_with_move(action)
                
                is_black = (self.env.board[action // BOARD_SIZE, action % BOARD_SIZE] == 1)
                self._sync_board_ui(new_action=action, is_black=is_black)
                
                # 💡 修復：繼承前一回合勝率，保持曲線連續
                last_wr = self.win_rate_history[-1] if self.win_rate_history else 0.5
                self.win_rate_history.append(last_wr)
                self.plot_curve.setData(self.win_rate_history)
                
                self._check_game_state()
                if not self.env.done:
                    if self._is_ai_turn():
                        self._trigger_ai_turn()
                    else:
                        is_black_next = (self.env.current_player == 1)
                        turn_str = "黑棋" if is_black_next else "白棋"
                        self.status_lbl.setText(f"等待 {turn_str} 落子")

    def shake_board(self, intensity=4, duration=150):
        self.board_shake_step = 0
        self.board_shake_max = duration // 15
        self.board_shake_intensity = intensity
        self.board_base_pos = self.view.pos()
        
        self.board_shake_timer = QTimer(self)
        self.board_shake_timer.timeout.connect(self._do_board_shake)
        self.board_shake_timer.start(15)
        
    def _do_board_shake(self):
        self.board_shake_step += 1
        if self.board_shake_step >= self.board_shake_max:
            self.board_shake_timer.stop()
            self.view.move(self.board_base_pos)
        else:
            dx = np.random.randint(-self.board_shake_intensity, self.board_shake_intensity + 1)
            dy = np.random.randint(-self.board_shake_intensity, self.board_shake_intensity + 1)
            self.view.move(self.board_base_pos.x() + dx, self.board_base_pos.y() + dy)

    def _sync_board_ui(self, new_action=None, is_black=True, is_fast_forward=False):
        if new_action is not None:
            r = new_action // BOARD_SIZE
            c = new_action % BOARD_SIZE
            piece = YunziPiece(r, c, is_black, is_new=not is_fast_forward)
            self.scene.addItem(piece)
            self.piece_items.append(piece)
            
            if not is_fast_forward:
                self.shake_board(intensity=4, duration=150)
            
            if hasattr(self, 'last_indicator') and self.last_indicator:
                if hasattr(self.last_indicator, 'timer'):
                    self.last_indicator.timer.stop()
                try:
                    self.scene.removeItem(self.last_indicator)
                except Exception:
                    pass
            
            self.last_indicator = LastIndicatorItem(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE)
            self.scene.addItem(self.last_indicator)

    def _trigger_ai_turn(self):
        self.ai_thinking = True
        self.slider_time.setEnabled(False)
        self.think_timer.start(300)
        self._worker_gen = getattr(self, '_worker_gen', 0) + 1
        current_gen = self._worker_gen
        self.worker = AIEngineWorker(self.env, self.model, self.device, self.mcts, self.current_diff, self.main_window.model_lock, worker_id=current_gen)
        self.worker.finished_signal.connect(self._on_ai_finished)
        self.worker.finished_signal.connect(self.worker.deleteLater)
        self.worker.start()

    def _on_ai_finished(self, action, win_rate, pv_path, acts, probs):
        self.think_timer.stop()
        self.ai_thinking = False
        self.slider_time.setEnabled(True)

        
        # 🔒 Stale Worker Guard: 若使用者已 reset/time-travel，忽略舊世代的結果
        worker = self.sender()
        if hasattr(worker, 'worker_id') and worker.worker_id != self._worker_gen:
            return
            
        # 🔒 時間線防禦：若歷史已被時光機預覽切斷，丟棄此落子
        if len(self.action_history) != self.slider_time.value():
            return
        
        self.action_history.append(action)
        self.slider_time.setMaximum(len(self.action_history))
        self.slider_time.setValue(len(self.action_history))
        
        self.env.step(action)
        self.mcts.update_with_move(action)
        
        is_black = (self.env.board[action // BOARD_SIZE, action % BOARD_SIZE] == 1)
        self._sync_board_ui(new_action=action, is_black=is_black)
        
        self.win_rate_history.append(win_rate)
        self.plot_curve.setData(self.win_rate_history)
        
        self.pv_log.append(f"> {' → '.join(pv_path)}\n")
        self.pv_log.verticalScrollBar().setValue(self.pv_log.verticalScrollBar().maximum())
        
        self.status_lbl.setText(f"黑棋勝率預估: {win_rate*100:.1f}%")
        self._check_game_state()
        
        if not self.env.done:
            if self._is_ai_turn():
                # 🔥 AI 互打模式或連續 AI 回合時，給予 600ms 動畫/視覺緩衝時間
                self.status_lbl.setText("AI 準備中...")
                QTimer.singleShot(600, self._trigger_ai_turn)
            else:
                self.status_lbl.setText("等待玩家落子")
                self.pv_log.append("👉 現在輪到您下棋了\n")
                
        cursor_pos = self.view.mapFromGlobal(QCursor.pos())
        if self.view.rect().contains(cursor_pos):
            self._handle_mouse_move(self.view.mapToScene(cursor_pos))

    def _check_game_state(self):
        if self.env.done:
            self._clear_ghost()
            
            if self.env.winner != 0:
                win_line = self._find_winning_line()
                if win_line:
                    self._play_win_animation(win_line)
            
            if self.env.winner == 0:
                self.status_lbl.setText("🤝 平局：勢均力敵")
            else:
                if self.game_mode == "PvP":
                    winner_str = "黑棋" if self.env.winner == 1 else "白棋"
                    self.status_lbl.setText(f"🎉 勝利：{winner_str} 獲勝！")
                    self.main_window.shake(intensity=15, duration=800)
                elif self.game_mode == "AIvAI":
                    winner_str = "黑棋" if self.env.winner == 1 else "白棋"
                    self.status_lbl.setText(f"🤖 模擬結束：{winner_str} 獲勝！")
                else:
                    if self.env.winner == self.human_player:
                        self.status_lbl.setText("🎉 勝利：人類超越了 AI")
                        self.main_window.shake(intensity=15, duration=800)
                    else:
                        self.status_lbl.setText("💀 失敗：您已被 AI 擊敗")
                        self.main_window.shake(intensity=25, duration=1000)

# ==========================================
# 🚀 應用程式啟動 (主控中心)
# ==========================================
class AlphaZeroApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero 對局終端")
        self.resize(1400, 950)
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(f"background-color: {APP_BG};")
        
        QApplication.setFont(QFont("Microsoft JhengHei", 12))
        self.model_lock = threading.Lock()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PolicyValueNet().to(self.device)
        self.model.eval()
        self._load_model()
        
        # 🚀 引入 torch.compile 進行圖編譯優化 (PyTorch 2.0+)，Windows 平台上停用以防 Triton 缺失崩潰
        if hasattr(torch, 'compile') and os.name != 'nt':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled using torch.compile.")
            except Exception as e:
                pass
        
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)
        
        self.menu_page = MainMenu(self)
        self.game_page = GameBoard(self)
        
        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.game_page)
        
        self.return_to_menu()
        
    def _load_model(self):
        model_path = './checkpoints/best_model.pth'
        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                # 🛡️ 容錯防禦：自動清洗 _orig_mod. 前綴，以相容舊版本存出的 compile 權重
                state_dict = ckpt['model_state_dict']
                clean_state_dict = {
                    (k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k): v
                    for k, v in state_dict.items()
                }
                self.model.load_state_dict(clean_state_dict)
                print("Model loaded successfully.")
            except Exception as e:
                print("Model load failed:", e)

    def start_game(self, game_mode, player_color, difficulty):
        self.game_page.setup_game(game_mode, player_color, difficulty)
        self.stack.setCurrentWidget(self.game_page)
        
    def return_to_menu(self):
        self.stack.setCurrentWidget(self.menu_page)

    def shake(self, intensity=10, duration=300):
        self.shake_anim_step = 0
        self.shake_max_steps = duration // 15
        self.shake_intensity = intensity
        self.base_pos = self.pos()
        
        self.shake_timer = QTimer(self)
        self.shake_timer.timeout.connect(self._do_shake)
        self.shake_timer.start(15)
        
    def _do_shake(self):
        self.shake_anim_step += 1
        if self.shake_anim_step >= self.shake_max_steps:
            self.shake_timer.stop()
            self.move(self.base_pos)
        else:
            dx = np.random.randint(-self.shake_intensity, self.shake_intensity)
            dy = np.random.randint(-self.shake_intensity, self.shake_intensity)
            self.move(self.base_pos.x() + dx, self.base_pos.y() + dy)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = AlphaZeroApp()
    window.show()
    sys.exit(app.exec())
