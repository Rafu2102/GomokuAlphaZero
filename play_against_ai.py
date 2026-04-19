import os
import sys
import time
import threading
import pygame
import torch
import numpy as np

# Suppress Pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from env import GomokuEnv
from mcts import MCTSEngine
from resnet import PolicyValueNet

# ==========================================
# Game Config & Styles
# ==========================================
WIDTH, HEIGHT = 750, 950
BOARD_SIZE = 15
MARGIN = 60
CELL_SIZE = (WIDTH - 2 * MARGIN) // (BOARD_SIZE - 1)

# Wood Theme Colors
WOOD_BASE = (222, 173, 107)
GRID_COLOR = (60, 30, 0)
HIGHLIGHT_COLOR = (255, 60, 80)
TEXT_COLOR = (50, 20, 0)

pygame.init()
pygame.display.set_caption("AlphaZero Gomoku - Mobile Edition")
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# 繁體中文支援 (微軟正黑體)
def get_chinese_font(size, bold=False):
    fonts = pygame.font.get_fonts()
    for f in ['microsoftjhenghei', 'simsun', 'simhei', 'dengxian']:
        if f in fonts:
            return pygame.font.SysFont(f, size, bold=bold)
    return pygame.font.SysFont(None, size)

font_large = get_chinese_font(54, bold=True)
font_medium = get_chinese_font(32, bold=True)
font_small = get_chinese_font(22)

# ==========================================
# Pre-render Assets (Mobile Game Look)
# ==========================================
def create_wood_background(w, h):
    surf = pygame.Surface((w, h))
    surf.fill(WOOD_BASE)
    
    # Procedural Wood Grain
    np.random.seed(42)  # 固定紋理
    for _ in range(500):
        c_val = max(120, min(255, WOOD_BASE[0] + np.random.randint(-20, 15)))
        color = (c_val, int(c_val * 0.78), int(c_val * 0.48))
        x = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        y2 = y1 + np.random.randint(50, 200)
        thick = np.random.randint(1, 3)
        pygame.draw.line(surf, color, (x, y1), (x, y2), thick)
        
    return surf

def create_piece_surface(color_type, radius):
    # Create surface larger than piece for drop shadow
    surf_size = int(radius * 3)
    surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
    center = (surf_size // 2, surf_size // 2)
    
    # 底部投影 (Drop Shadow)
    pygame.draw.circle(surf, (0, 0, 0, 120), (center[0] + 4, center[1] + 6), radius)
    
    if color_type == 'black':
        # 黑子主體
        pygame.draw.circle(surf, (25, 25, 25), center, radius)
        # 高光反射 (質感)
        pygame.draw.circle(surf, (90, 90, 90), (center[0] - radius//3, center[1] - radius//3), radius//3)
    else:
        # 白子主體
        pygame.draw.circle(surf, (245, 245, 245), center, radius)
        # 邊緣陰影 (立體感)
        pygame.draw.circle(surf, (200, 200, 200), center, radius, 2)
        # 高光反射
        pygame.draw.circle(surf, (255, 255, 255), (center[0] - radius//3, center[1] - radius//3), radius//3)

    return surf

# Generate static assets taking up very little loading time
bg_surface = create_wood_background(WIDTH, HEIGHT)
black_piece_img = create_piece_surface('black', CELL_SIZE // 2 - 2)
white_piece_img = create_piece_surface('white', CELL_SIZE // 2 - 2)

# ==========================================
# Engine Initialization
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PolicyValueNet().to(device)
model.eval()

model_path = './checkpoints/best_model.pth'
model_loaded = False
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_loaded = True
    except:
        pass

def predict_fn(state_tensor):
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            t = torch.from_numpy(state_tensor).unsqueeze(0).to(device)
            log_probs, value = model(t)
        probs = torch.exp(log_probs).cpu().numpy()[0]
        v = float(value.cpu().numpy()[0][0])
    return list(enumerate(probs)), v


# ==========================================
# Graphics Rendering Loop
# ==========================================
def draw_board(env, hover_x=None, hover_y=None):
    # 畫出木頭背景
    screen.blit(bg_surface, (0, 0))
    
    # 頂部狀態列
    if env.done:
        status_text = "遊戲結束"
        st_color = HIGHLIGHT_COLOR
    elif ai_thinking:
        status_text = "機器對手思考中..."
        st_color = (200, 10, 10)
    else:
        status_text = "換你下棋"
        st_color = TEXT_COLOR
        
    txt_surf = font_medium.render(status_text, True, st_color)
    screen.blit(txt_surf, (WIDTH // 2 - txt_surf.get_width() // 2, 30))

    offset_y = 120
    
    # 繪製網格
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, GRID_COLOR, 
                         (MARGIN, MARGIN + i * CELL_SIZE + offset_y), 
                         (WIDTH - MARGIN, MARGIN + i * CELL_SIZE + offset_y), 2)
        pygame.draw.line(screen, GRID_COLOR, 
                         (MARGIN + i * CELL_SIZE, MARGIN + offset_y), 
                         (MARGIN + i * CELL_SIZE, WIDTH - MARGIN + offset_y), 2)
                         
    # 天元與星位 (五子棋標準 5 個)
    stars = [(3,3), (11,3), (3,11), (11,11), (7,7)]
    for r, c in stars:
        pygame.draw.circle(screen, GRID_COLOR, 
                           (MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE + offset_y), 5)

    # 繪製棋子
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if env.board[r, c] != 0:
                is_black = (env.board[r, c] == 1)
                img = black_piece_img if is_black else white_piece_img
                
                # 計算置中 (扣掉預渲染放大比例的偏差)
                pos = (MARGIN + c * CELL_SIZE - img.get_width()//2, 
                       MARGIN + r * CELL_SIZE + offset_y - img.get_height()//2)
                
                screen.blit(img, pos)
                
                # Highlight last move
                if (r * BOARD_SIZE + c) == env.last_move:
                    center_pos = (MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE + offset_y)
                    pygame.draw.circle(screen, HIGHLIGHT_COLOR, center_pos, 6)
                    pygame.draw.circle(screen, (255, 255, 255), center_pos, 3)

    # Hover 效果 (半透明棋子)
    if hover_x is not None and hover_y is not None and not ai_thinking and not env.done:
        if env.board[hover_x, hover_y] == 0:
            is_black = (env.current_player == 1)
            img = black_piece_img.copy() if is_black else white_piece_img.copy()
            img.set_alpha(150) # 半透明
            pos = (MARGIN + hover_y * CELL_SIZE - img.get_width()//2, 
                   MARGIN + hover_x * CELL_SIZE + offset_y - img.get_height()//2)
            screen.blit(img, pos)
            
    # 底部資訊
    info = f"Engine: {'Online (Ready)' if model_loaded else 'Offline (Random Agent)'}"
    info_surf = font_small.render(info, True, (100, 60, 30))
    screen.blit(info_surf, (20, HEIGHT - 40))

# ==========================================
# ASYNC Logic
# ==========================================
env = GomokuEnv()
mcts = None

STATE_MENU = 0
STATE_PLAYING = 1
STATE_GAMEOVER = 2
current_state = STATE_MENU
human_player = 1

ai_thinking = False
ai_move_action = None

def ai_compute_thread():
    global ai_move_action, ai_thinking
    time.sleep(0.3) # 模擬一點點人類思考感
    acts, probs = mcts.get_action_probs(env, predict_fn, temperature=1e-3)
    ai_move_action = acts[np.argmax(probs)]
    ai_thinking = False

def restart_game(player_choice):
    global env, mcts, current_state, human_player, ai_thinking, ai_move_action
    human_player = player_choice
    env.reset()
    mcts = MCTSEngine(c_puct=2.0, n_playout=400)
    current_state = STATE_PLAYING
    ai_thinking = False
    ai_move_action = None
    
    if human_player == -1: # AI 當黑子先下
        ai_thinking = True
        threading.Thread(target=ai_compute_thread, daemon=True).start()

# ==========================================
# Main Game Loop
# ==========================================
running = True
clock = pygame.time.Clock()
hover_r, hover_c = None, None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        mouse_pos = pygame.mouse.get_pos()
        
        # [ MENU STATE ]
        if current_state == STATE_MENU:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if HEIGHT//2 <= mouse_pos[1] <= HEIGHT//2 + 70:
                    restart_game(1)  # 玩家黑
                elif HEIGHT//2 + 90 <= mouse_pos[1] <= HEIGHT//2 + 160:
                    restart_game(-1) # 玩家白
                    
        # [ PLAYING STATE ]
        elif current_state == STATE_PLAYING:
            offset_y = 120
            # 檢查滑鼠是否在棋盤內
            if (MARGIN - CELL_SIZE//2 <= mouse_pos[0] <= WIDTH - MARGIN + CELL_SIZE//2 and 
                MARGIN + offset_y - CELL_SIZE//2 <= mouse_pos[1] <= WIDTH - MARGIN + offset_y + CELL_SIZE//2):
                
                hover_c = round((mouse_pos[0] - MARGIN) / CELL_SIZE)
                hover_r = round((mouse_pos[1] - MARGIN - offset_y) / CELL_SIZE)
                hover_c = max(0, min(BOARD_SIZE-1, hover_c))
                hover_r = max(0, min(BOARD_SIZE-1, hover_r))
            else:
                hover_r, hover_c = None, None

            # 處理人類點擊
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not ai_thinking and env.current_player == human_player and hover_r is not None:
                    action = hover_r * BOARD_SIZE + hover_c
                    if action in env.get_legal_moves():
                        env.step(action)
                        mcts.update_with_move(action)
                        
                        if not env.done:
                            ai_thinking = True
                            threading.Thread(target=ai_compute_thread, daemon=True).start()
                            
        # [ GAMEOVER STATE ]
        elif current_state == STATE_GAMEOVER:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                current_state = STATE_MENU

    # 處理 AI 推演完畢的狀態同步
    if current_state == STATE_PLAYING:
        if not ai_thinking and ai_move_action is not None:
            env.step(ai_move_action)
            mcts.update_with_move(ai_move_action)
            ai_move_action = None
            
        if env.done:
            current_state = STATE_GAMEOVER

    # 畫面繪製
    if current_state == STATE_MENU:
        screen.blit(bg_surface, (0, 0))
        
        # 半透明底板遮罩
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        screen.blit(overlay, (0, 0))
        
        title = font_large.render("AlphaZero 五子棋", True, (255, 255, 255))
        screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//3 - 50))
        
        # 執黑按鈕
        btn1 = pygame.Rect(WIDTH//2 - 180, HEIGHT//2, 360, 70)
        pygame.draw.rect(screen, (30, 30, 30), btn1, border_radius=35)
        pygame.draw.rect(screen, (255, 255, 255), btn1, width=2, border_radius=35)
        t1 = font_medium.render("玩家執黑 〇 (先手)", True, (240, 240, 240))
        screen.blit(t1, (WIDTH//2 - t1.get_width()//2, HEIGHT//2 + 15))
        
        # 執白按鈕
        btn2 = pygame.Rect(WIDTH//2 - 180, HEIGHT//2 + 90, 360, 70)
        pygame.draw.rect(screen, (240, 240, 240), btn2, border_radius=35)
        pygame.draw.rect(screen, (30, 30, 30), btn2, width=2, border_radius=35)
        t2 = font_medium.render("玩家執白 Ｘ (後手)", True, (30, 30, 30))
        screen.blit(t2, (WIDTH//2 - t2.get_width()//2, HEIGHT//2 + 105))
        
    elif current_state in [STATE_PLAYING, STATE_GAMEOVER]:
        draw_board(env, hover_r, hover_c)
        
        if current_state == STATE_GAMEOVER:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            
            if env.winner == human_player:
                msg = "你贏了！"
                c = (150, 255, 150)
            elif env.winner == -human_player:
                msg = "AI 獲勝！"
                c = HIGHLIGHT_COLOR
            else:
                msg = "平局！"
                c = (255, 255, 255)
                
            txt = font_large.render(msg, True, c)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 60))
            
            sub = font_small.render("點擊螢幕任意處返回主選單", True, (200, 200, 200))
            screen.blit(sub, (WIDTH//2 - sub.get_width()//2, HEIGHT//2 + 40))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
