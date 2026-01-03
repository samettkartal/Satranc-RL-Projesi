# DOSYA: bot_vs_bot.py
# Açıklama: Kendi eğittiğimiz RL modelimiz (AlphaZero benzeri) ile
# Stockfish motorunun kapışmasını GÖRSEL olarak izlemek için
# otomatik bir betik.

import pygame
import chess
import torch
import threading
import queue
import os
import sys
import random # Tarafları rastgele belirlemek için
from collections import Counter
from stockfish import Stockfish

# Kendi motorumuzun bileşenlerini içe aktaralım
from rl_chess import config, network, mcts, utils

# --- Arayüz Ayarları (Değişmedi) ---
BOARD_WIDTH = 640
INFO_PANEL_WIDTH = 200
SCREEN_WIDTH = BOARD_WIDTH + INFO_PANEL_WIDTH
SCREEN_HEIGHT = BOARD_WIDTH
SQUARE_SIZE = BOARD_WIDTH // 8
MINI_SQUARE_SIZE = INFO_PANEL_WIDTH // 6 
BOARD_LIGHT_COLOR = (234, 221, 197)
BOARD_DARK_COLOR = (168, 136, 101)
INFO_PANEL_BG = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)
GAMEOVER_OVERLAY_COLOR = (0, 0, 0, 180)
GAMEOVER_BOX_COLOR = (230, 230, 230)
GAMEOVER_TEXT_COLOR = (10, 10, 10)

# --- Puanlama ve Diğer Sabitler (Değişmedi) ---
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}
INITIAL_PIECE_COUNTS = {
    'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1, 'K': 1,
    'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1, 'k': 1
}
PIECE_TO_FILENAME = {
    'P': 'wP.png', 'N': 'wN.png', 'B': 'wB.png', 'R': 'wR.png', 'Q': 'wQ.png', 'K': 'wK.png',
    'p': 'bP.png', 'n': 'bN.png', 'b': 'bB.png', 'r': 'bR.png', 'q': 'bQ.png', 'k': 'bK.png'
}
CAPTURES_BG_FILENAME = "captures_bg.png"

# --- Motor Ayarları ---
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"
STOCKFISH_DIFFICULTY = 10 # 0-20 arası (Stockfish'in zorluğu)
RL_MODEL_SIMULATIONS = 400 # Kendi modelimizin hamle başına düşünme sayısı

# Hamleler arası bekleme süresi (izleyebilmek için)
MOVE_DELAY_MS = 500 # 0.5 saniye

class BotVsBotGUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("KAPANIŞ: RL Model vs Stockfish")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("Arial", 30, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 20)
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_gameover_title = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_gameover_sub = pygame.font.SysFont("Arial", 18)

        # --- 1. Motorları ve Görselleri Yükle ---
        self.rl_engine = self.load_rl_engine()
        self.stockfish_engine = self.load_stockfish_engine()
        if self.rl_engine is None or self.stockfish_engine is None:
            print("HATA: Motorlardan biri yüklenemedi. Betik durduruluyor.")
            sys.exit(1)
            
        self.assets = self.load_assets()
        if not self.assets.get('P'):
            print("HATA: 'assets' klasöründeki taş resimleri yüklenemedi.")
            sys.exit(1)
        
        # --- 2. Oyun Durumu Değişkenleri ---
        self.board = chess.Board()
        self.game_over = False
        self.engines = {} # {chess.WHITE: engine, chess.BLACK: engine}
        self.bot_move_queue = queue.Queue() # Düşünme thread'i hamleyi buraya atar
        self.last_move_time = 0 # Hamleler arası gecikme için
        self.waiting_for_move = False # Zaten bir hamle bekleniyor mu?
        
        # Puanlama
        self.captured_by_white = [] 
        self.captured_by_black = [] 
        self.material_advantage = 0 
        
        # --- 3. Oyunu Başlat ---
        self.start_new_game()

    def load_rl_engine(self):
        """Kendi eğittiğimiz RL (MCTS) motorumuzu yükler."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        model_path = os.path.join(models_dir, config.BEST_MODEL_NAME)
        
        if not os.path.exists(model_path): 
            print(f"HATA: RL Modeli bulunamadı: {model_path}")
            return None
        try:
            model = network.PolicyValueNet().to(config.DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            model.eval()
            
            mcts_search = mcts.MCTS(model) # MCTS motorunu modelle başlat
            
            print(f"RL Modeli (MCTS) başarıyla yüklendi: {model_path}")
            return mcts_search # Dönen nesne MCTS motorudur
        except Exception as e:
            print(f"HATA: RL Modeli yüklenemedi! {e}")
            return None

    def load_stockfish_engine(self):
        """Stockfish motorunu yükler."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(script_dir, STOCKFISH_PATH)
        
        if not os.path.exists(engine_path): 
            print(f"HATA: Stockfish motoru bulunamadı: {engine_path}")
            return None
        try:
            stockfish = Stockfish(path=engine_path, parameters={
                "Skill Level": STOCKFISH_DIFFICULTY
            })
            print(f"Stockfish motoru başarıyla yüklendi: {engine_path}")
            return stockfish
        except Exception as e:
            print(f"HATA: Stockfish yüklenemedi! {e}")
            return None

    def load_assets(self):
        """'assets' klasöründeki tüm görselleri yükler."""
        # (Bu fonksiyon, gui_pygame_v3'teki DÜZELTİLMİŞ KOD ile aynıdır)
        images = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_path = os.path.join(script_dir, "assets")
        if not os.path.exists(assets_path):
            print(f"HATA: 'assets' klasörü bulunamadı! Aranan yer: {assets_path}")
            return {}
        for symbol, filename in PIECE_TO_FILENAME.items():
            path = os.path.join(assets_path, filename)
            if not os.path.exists(path):
                print(f"HATA (Durum 1): Dosya bulunamadı! -> {path}")
                return {}
            try:
                img = pygame.image.load(path).convert_alpha()
                if img is None: raise Exception("Pygame resmi yükleyemedi (None döndürdü).")
                images[symbol] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
                images[f"{symbol}_mini"] = pygame.transform.smoothscale(img, (MINI_SQUARE_SIZE, MINI_SQUARE_SIZE))
            except Exception as e: 
                print(f"HATA (Durum 2): Dosya bozuk veya okunamıyor! -> {path} (Hata: {e})")
                return {}
        bg_path = os.path.join(assets_path, CAPTURES_BG_FILENAME)
        if os.path.exists(bg_path):
            try:
                images['captures_bg'] = pygame.image.load(bg_path).convert_alpha()
            except Exception: images['captures_bg'] = None
        else: images['captures_bg'] = None
        print("Görseller (büyük ve minik) başarıyla yüklendi.")
        return images

    #
    # --- Çizim Fonksiyonları (draw_board, draw_pieces, draw_info_panel...) ---
    # --- gui_pygame_v3'ten KOPYALANDI, DEĞİŞİKLİK YOK ---
    #
    def draw_board(self):
        for r in range(8):
            for c in range(8):
                color = BOARD_LIGHT_COLOR if (r + c) % 2 == 0 else BOARD_DARK_COLOR
                pygame.draw.rect(self.screen, color, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def draw_pieces(self):
        for r in range(8):
            for c in range(8):
                square = chess.square(file_index=c, rank_index=7 - r)
                piece = self.board.piece_at(square)
                if piece:
                    symbol = piece.symbol()
                    if symbol in self.assets:
                        img = self.assets[symbol]
                        self.screen.blit(img, (c * SQUARE_SIZE, r * SQUARE_SIZE))

    def draw_info_panel(self):
        panel_rect = (BOARD_WIDTH, 0, INFO_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, INFO_PANEL_BG, panel_rect)
        adv_text = "0"
        if self.material_advantage > 0: adv_text = f"+{self.material_advantage}"
        elif self.material_advantage < 0: adv_text = f"{self.material_advantage}"
        adv_surf = self.font_large.render(adv_text, True, TEXT_COLOR)
        adv_rect = adv_surf.get_rect(center=(BOARD_WIDTH + INFO_PANEL_WIDTH // 2, 40))
        self.screen.blit(adv_surf, adv_rect)
        white_cap_bg_rect = pygame.Rect(BOARD_WIDTH + 10, 105, INFO_PANEL_WIDTH - 20, 200)
        black_cap_bg_rect = pygame.Rect(BOARD_WIDTH + 10, 345, INFO_PANEL_WIDTH - 20, 200)
        # Sadece düz renk kutular çiz (Resim yok)
        pygame.draw.rect(self.screen, (30,30,30), white_cap_bg_rect); pygame.draw.rect(self.screen, (30,30,30), black_cap_bg_rect)

        y_offset = 80; x_offset = BOARD_WIDTH + 10
        bot_captures_surf = self.font_small.render("Stockfish'in Aldıkları:", True, (180,180,180))
        if self.engines.get(chess.WHITE) == self.rl_engine:
             bot_captures_surf = self.font_small.render("RL Modelin Aldıkları:", True, (180,180,180))
        self.screen.blit(bot_captures_surf, (x_offset, y_offset))
        # y_offset = 110 # Resimler kaldırıldı

        y_offset = 320; x_offset = BOARD_WIDTH + 10
        human_captures_surf = self.font_small.render("RL Modelin Aldıkları:", True, (180,180,180))
        if self.engines.get(chess.WHITE) == self.rl_engine:
            human_captures_surf = self.font_small.render("Stockfish'in Aldıkları:", True, (180,180,180))
        self.screen.blit(human_captures_surf, (x_offset, y_offset))
        # y_offset = 350 # Resimler kaldırıldı

    def draw_game_over_screen(self):
        if not self.game_over: return
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill(GAMEOVER_OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))
        card_width, card_height = 400, 200
        card_x = (SCREEN_WIDTH - card_width) // 2; card_y = (SCREEN_HEIGHT - card_height) // 2
        card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
        pygame.draw.rect(self.screen, GAMEOVER_BOX_COLOR, card_rect, border_radius=10)
        result = self.board.result(claim_draw=True)
        # Hangi motorun kazandığını belirle
        winner_name = None
        if result == '1-0': # Beyaz kazandı
            winner_name = "RL Model" if isinstance(self.engines[chess.WHITE], mcts.MCTS) else "Stockfish"
            msg = f"OYUN BİTTİ! {winner_name} (Beyaz) Kazandı!"
        elif result == '0-1': # Siyah kazandı
            winner_name = "RL Model" if isinstance(self.engines[chess.BLACK], mcts.MCTS) else "Stockfish"
            msg = f"OYUN BİTTİ! {winner_name} (Siyah) Kazandı!"
        else: msg = "OYUN BİTTİ! Berabere."
        title_surf = self.font_gameover_title.render(msg, True, GAMEOVER_TEXT_COLOR, wraplength=card_width - 20)
        title_rect = title_surf.get_rect(center=(card_rect.centerx, card_rect.centery - 30))
        self.screen.blit(title_surf, title_rect)
        sub_surf = self.font_gameover_sub.render("Yeni kapışma için 'N' tuşuna basın.", True, (80, 80, 80))
        sub_rect = sub_surf.get_rect(center=(card_rect.centerx, card_rect.centery + 60))
        self.screen.blit(sub_surf, sub_rect)

    #
    # --- Oyun Mantığı Fonksiyonları (Botlar için güncellendi) ---
    #
    def update_captures_and_score(self):
        current_counts = Counter(p.symbol() for p in self.board.piece_map().values())
        self.captured_by_white = []; self.captured_by_black = []
        for symbol, initial_count in INITIAL_PIECE_COUNTS.items():
            current_count = current_counts.get(symbol, 0)
            diff = initial_count - current_count
            if diff > 0:
                piece = chess.Piece.from_symbol(symbol)
                for _ in range(diff):
                    if piece.color == chess.BLACK: self.captured_by_white.append(piece.piece_type)
                    else: self.captured_by_black.append(piece.piece_type)
        self.white_score = sum(PIECE_VALUES.get(pt, 0) for pt in self.captured_by_white)
        self.black_score = sum(PIECE_VALUES.get(pt, 0) for pt in self.captured_by_black)
        self.material_advantage = self.white_score - self.black_score
            
    def trigger_bot_move(self):
        """Sırası gelen botun düşünme thread'ini tetikler."""
        if self.game_over or self.waiting_for_move:
            return
            
        self.waiting_for_move = True # Yeni hamle istemeden önce mevcut olanı bekle
        
        # Sırası gelen motoru al
        engine_to_run = self.engines[self.board.turn]
        
        # Hangi motor olduğuna karar ver ve doğru thread'i başlat
        if isinstance(engine_to_run, mcts.MCTS):
            # Bu bizim RL Modelimiz
            threading.Thread(target=self.rl_think_thread, daemon=True).start()
        elif isinstance(engine_to_run, Stockfish):
            # Bu Stockfish
            threading.Thread(target=self.stockfish_think_thread, daemon=True).start()

    def rl_think_thread(self):
        """[ARKA PLAN THREAD'İ] Bizim RL modelimiz için."""
        try:
            original_sims = config.MCTS_SIMULATIONS
            config.MCTS_SIMULATIONS = RL_MODEL_SIMULATIONS
            
            ai_move, _, _ = self.rl_engine.get_best_move(self.board, temperature=0.0)
            
            config.MCTS_SIMULATIONS = original_sims
            self.bot_move_queue.put(ai_move)
        except Exception as e:
            print(f"HATA: RL Model (MCTS) thread'i çöktü: {e}")
            self.bot_move_queue.put(None)

    def stockfish_think_thread(self):
        """[ARKA PLAN THREAD'İ] Stockfish için."""
        try:
            self.stockfish_engine.set_fen_position(self.board.fen())
            # 1 saniye (1000ms) düşünme süresi
            best_move_uci = self.stockfish_engine.get_best_move_time(1000)
            
            if best_move_uci:
                ai_move = chess.Move.from_uci(best_move_uci)
                self.bot_move_queue.put(ai_move)
            else:
                self.bot_move_queue.put(None)
        except Exception as e:
            print(f"HATA: Stockfish düşünme thread'i çöktü: {e}")
            self.bot_move_queue.put(None)

    def check_bot_queue(self):
        """Hamle kuyruğunu kontrol eder ve hamleyi uygular."""
        if self.game_over:
            return
            
        try:
            ai_move = self.bot_move_queue.get_nowait()
            
            if ai_move and ai_move in self.board.legal_moves:
                self.board.push(ai_move)
                self.update_captures_and_score()
                print(f"Oynanan Hamle: {ai_move.uci()} (Sıra: {'Beyaz' if self.board.turn == chess.WHITE else 'Siyah'})")
                
                # Botun hamlesinden sonra bir gecikme ver
                self.last_move_time = pygame.time.get_ticks() 
                self.waiting_for_move = False # Artık yeni bir hamle isteyebiliriz
                
                self.check_game_over()
            else: 
                if ai_move is not None:
                    print(f"HATA: Bot yasal olmayan hamle döndürdü: {ai_move.uci()}")
                self.waiting_for_move = False # Hata olsa bile devam et

        except queue.Empty: 
            pass # Kuyruk boşsa (bot hala düşünüyorsa) sorun yok

    def check_game_over(self):
        if self.board.is_game_over(claim_draw=True) and not self.game_over:
            self.game_over = True
            print("Oyun Bitti.")
            return True
        return False

    def start_new_game(self):
        """Yeni bir kapışma başlatır ve tarafları rastgele atar."""
        self.board.reset()
        self.game_over = False
        self.captured_by_white = []
        self.captured_by_black = []
        self.update_captures_and_score()
        while not self.bot_move_queue.empty():
            self.bot_move_queue.get()
            
        self.stockfish_engine.set_fen_position(chess.STARTING_FEN)
        
        # --- YENİ: Tarafları Rastgele Belirle ---
        self.engines = {}
        if random.choice([True, False]):
            self.engines[chess.WHITE] = self.rl_engine
            self.engines[chess.BLACK] = self.stockfish_engine
            print("\n--- YENİ KAPIŞMA (Beyaz: RL Model | Siyah: Stockfish) ---")
        else:
            self.engines[chess.WHITE] = self.stockfish_engine
            self.engines[chess.BLACK] = self.rl_engine
            print("\n--- YENİ KAPIŞMA (Beyaz: Stockfish | Siyah: RL Model) ---")
            
        self.last_move_time = pygame.time.get_ticks()
        self.waiting_for_move = False
        # self.trigger_bot_move() # İlk hamleyi main_loop tetiklesin

    def main_loop(self):
        """Ana Pygame döngüsü."""
        running = True
        while running:
            # --- Olay (Event) Yönetimi ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # 'N' tuşu ile yeni kapışma
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        self.start_new_game()
                
                # Oyun bittiyse tıkla -> yeni kapışma
                if event.type == pygame.MOUSEBUTTONDOWN and self.game_over:
                    self.start_new_game()

            # --- Botların Hamle Mantığı ---
            current_time = pygame.time.get_ticks()
            if not self.game_over and not self.waiting_for_move:
                # Bir sonraki hamleyi tetiklemeden önce MOVE_DELAY_MS kadar bekle
                if current_time - self.last_move_time > MOVE_DELAY_MS:
                    self.trigger_bot_move()
            
            # Kuyruktan gelen hamleyi uygula (her frame kontrol et)
            self.check_bot_queue()

            # --- Çizim (Drawing) ---
            self.draw_board()
            self.draw_info_panel() 
            self.draw_pieces()
            
            if self.game_over:
                self.draw_game_over_screen() 

            # --- Ekranı Güncelle ---
            pygame.display.flip()
            self.clock.tick(30) # 30 FPS

        pygame.quit()
        sys.exit()

# --- Uygulamayı Başlatma ---
if __name__ == "__main__":
    app = BotVsBotGUI()
    app.main_loop()