# DOSYA: gui_pygame.py (NİHAİ DÜZELTİLMİŞ KOD)
# Açıklama: Pygame arayüzü
# Düzeltmeler:
# 1. 'assets' yolu %100 doğru çalışacak (os.path.dirname)
# 2. BOZUK PNG'leri yakalamak için try...except Exception eklendi.

import pygame
import chess
import torch
import threading
import queue
import os
import sys
from collections import Counter

# Projemizin çekirdek bileşenlerini içe aktaralım
from rl_chess import config, network, mcts, utils

# --- Arayüz Ayarları ---
BOARD_WIDTH = 640  # 8x8 Tahta
INFO_PANEL_WIDTH = 200 # Sağdaki bilgi paneli
SCREEN_WIDTH = BOARD_WIDTH + INFO_PANEL_WIDTH # Toplam 840
SCREEN_HEIGHT = BOARD_WIDTH # Toplam 640
SQUARE_SIZE = BOARD_WIDTH // 8
MINI_SQUARE_SIZE = INFO_PANEL_WIDTH // 6 

# Renkler
BOARD_LIGHT_COLOR = (234, 221, 197) # (R, G, B)
BOARD_DARK_COLOR = (168, 136, 101)
INFO_PANEL_BG = (40, 40, 40) # Koyu panel
SELECTED_COLOR = (247, 196, 77, 150) # (R, G, B, Alpha - yarı saydam)
LEGAL_MOVE_COLOR = (20, 80, 20, 100) # Yasal hamleler için
TEXT_COLOR = (255, 255, 255)
GAMEOVER_OVERLAY_COLOR = (0, 0, 0, 180) # Oyun sonu karartması
GAMEOVER_BOX_COLOR = (230, 230, 230) # Oyun sonu kartı
GAMEOVER_TEXT_COLOR = (10, 10, 10)

GUI_MCTS_SIMULATIONS = 400 

# --- Puanlama ---
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

# Terfi için semboller (İnsan her zaman Beyazdır)
PROMOTION_PIECES_SYMBOLS = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
CAPTURES_BG_FILENAME = "captures_bg.png"


class ChessGUI_Pygame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("RL Satranç Botu")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("Arial", 30, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 20)
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_gameover_title = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_gameover_sub = pygame.font.SysFont("Arial", 18)

        # --- 1. Modeli ve Görselleri Yükle ---
        self.model = self.load_bot_model()
        if self.model is None:
            print("HATA: Model yüklenemedi. 'main.py'yi çalıştırdınız mı?")
            sys.exit(1)
            
        self.assets = self.load_assets()
        # --- DÜZELTME: Anahtar 'wP' değil, 'P' (Piyon sembolü) olmalı ---
        if not self.assets.get('P'): # Yüklemenin başarılı olduğunu kontrol et
            print("\n!!! YÜKLEME BAŞARISIZ !!!")
            print("\n!!! YÜKLEME BAŞARISIZ !!!")
            print("HATA: 'assets' klasöründeki taş resimleri yüklenemedi.")
            print("Lütfen yukarıdaki 'HATA (Durum 1)' veya 'HATA (Durum 2)' mesajını kontrol edin.")
            sys.exit(1)

        self.mcts_search = mcts.MCTS(self.model)
        
        # --- 2. Oyun Durumu Değişkenleri ---
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.human_turn = True
        self.game_over = False
        self.captured_by_white = [] 
        self.captured_by_black = [] 
        self.material_advantage = 0 
        self.promotion_pending = False
        self.promotion_move_coords = None
        self.promotion_choice_rects = {}
        
        # --- 3. Thread Yönetimi ---
        self.bot_move_queue = queue.Queue()

    def load_bot_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        model_path = os.path.join(models_dir, config.BEST_MODEL_NAME)
        
        if not os.path.exists(model_path): 
            print(f"HATA: Model dosyası beklenen yolda bulunamadı: {model_path}")
            return None
        try:
            model = network.PolicyValueNet().to(config.DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            model.eval()
            print(f"Model başarıyla yüklendi: {model_path}")
            return model
        except Exception as e:
            print(f"HATA: Model yüklenemedi! {e}")
            return None

    def load_assets(self):
        """'assets' klasöründeki tüm görselleri (taşlar + arka plan) yükler."""
        images = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_path = os.path.join(script_dir, "assets")
        
        if not os.path.exists(assets_path):
            print(f"HATA: 'assets' klasörü bulunamadı! Aranan yer: {assets_path}")
            return {}

        # 1. Taşları yükle (büyük ve minik)
        for symbol, filename in PIECE_TO_FILENAME.items():
            path = os.path.join(assets_path, filename)
            
            if not os.path.exists(path):
                print(f"HATA (Durum 1): Dosya bulunamadı! -> {path}")
                return {}
            
            # --- DÜZELTME BAŞLANGICI (TÜM Hataları Yakala) ---
            try:
                img = pygame.image.load(path)
                if img is None:
                    raise Exception("Pygame resmi yükleyemedi (None döndürdü).")
                    
                img = img.convert_alpha()
                if img is None:
                    raise Exception("Resim 'convert_alpha()' yapılamadı.")

                images[symbol] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
                images[f"{symbol}_mini"] = pygame.transform.smoothscale(img, (MINI_SQUARE_SIZE, MINI_SQUARE_SIZE))
            
            except Exception as e: 
                print("="*50)
                print(f"!!! HATA (Durum 2): BİR DOSYA YÜKLENEMEDİ !!!")
                print(f"Hata yaşanan dosya: {path}")
                print(f"Hata mesajı: {e}")
                print("Bu, genellikle dosyanın bozuk, 0 byte veya geçersiz bir PNG olduğu anlamına gelir.")
                print("Lütfen bu PNG dosyasını silip 'assets' klasörüne yeniden indirin.")
                print("="*50)
                return {} # Hatalı dosya varsa başarısız say
            # --- DÜZELTME SONU ---
        
        # 2. Yenilen taşlar için arka planı yükle (isteğe bağlı)
        bg_path = os.path.join(assets_path, CAPTURES_BG_FILENAME)
        if os.path.exists(bg_path):
            try:
                images['captures_bg'] = pygame.image.load(bg_path).convert_alpha()
                print(f"'{CAPTURES_BG_FILENAME}' arka planı başarıyla yüklendi.")
            except Exception as e:
                print(f"Uyarı: '{CAPTURES_BG_FILENAME}' bozuk veya okunamıyor. Düz renk kullanılacak. Hata: {e}")
                images['captures_bg'] = None
        else:
            images['captures_bg'] = None
            print(f"Uyarı: '{CAPTURES_BG_FILENAME}' bulunamadı. Düz renk kullanılacak.")

        print("Görseller (büyük ve minik) başarıyla yüklendi.")
        return images

    def draw_board(self):
        for r in range(8):
            for c in range(8):
                color = BOARD_LIGHT_COLOR if (r + c) % 2 == 0 else BOARD_DARK_COLOR
                pygame.draw.rect(
                    self.screen, color, 
                    (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )

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

    def draw_highlights(self):
        if self.selected_square is not None and not self.promotion_pending:
            c = chess.square_file(self.selected_square)
            r = 7 - chess.square_rank(self.selected_square)
            highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surf.fill(SELECTED_COLOR)
            self.screen.blit(highlight_surf, (c * SQUARE_SIZE, r * SQUARE_SIZE))
            
            for move in self.legal_moves_for_selected:
                to_c = chess.square_file(move.to_square)
                to_r = 7 - chess.square_rank(move.to_square)
                move_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                move_surf.fill(LEGAL_MOVE_COLOR)
                self.screen.blit(move_surf, (to_c * SQUARE_SIZE, to_r * SQUARE_SIZE))

    def draw_info_panel(self):
        panel_rect = (BOARD_WIDTH, 0, INFO_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, INFO_PANEL_BG, panel_rect)

        # --- Puanlama ---
        adv_text = "0"
        if self.material_advantage > 0: adv_text = f"+{self.material_advantage}"
        elif self.material_advantage < 0: adv_text = f"{self.material_advantage}"
        adv_surf = self.font_large.render(adv_text, True, TEXT_COLOR)
        adv_rect = adv_surf.get_rect(center=(BOARD_WIDTH + INFO_PANEL_WIDTH // 2, 40))
        self.screen.blit(adv_surf, adv_rect)

        # --- Arka Planları Çiz ---
        white_cap_bg_rect = pygame.Rect(BOARD_WIDTH + 10, 105, INFO_PANEL_WIDTH - 20, 200)
        black_cap_bg_rect = pygame.Rect(BOARD_WIDTH + 10, 345, INFO_PANEL_WIDTH - 20, 200)
        
        # Sadece düz renk kutular çiz (Resim yok)
        pygame.draw.rect(self.screen, (30,30,30), white_cap_bg_rect)
        pygame.draw.rect(self.screen, (30,30,30), black_cap_bg_rect)

        # --- Botun Yediği Taşlar (Beyaz Taşlar) ---
        y_offset = 80
        x_offset = BOARD_WIDTH + 10
        bot_captures_surf = self.font_small.render("Botun Aldıkları:", True, (180,180,180))
        self.screen.blit(bot_captures_surf, (x_offset, y_offset))
        # y_offset = 110 # Resimler kaldırıldı

        # --- İnsanın Yediği Taşlar (Siyah Taşlar) ---
        y_offset = 320 
        x_offset = BOARD_WIDTH + 10
        human_captures_surf = self.font_small.render("Sizin Aldıklarınız:", True, (180,180,180))
        self.screen.blit(human_captures_surf, (x_offset, y_offset))
        # y_offset = 350 # Resimler kaldırıldı

    def draw_promotion_choice(self):
            """Piyon terfisi için seçim kutusunu çizer."""
            if not self.promotion_pending:
                return
                
            from_sq, to_sq = self.promotion_move_coords
            to_c = chess.square_file(to_sq)
            to_r = 7 - chess.square_rank(to_sq) # Pygame satırı
            
            if to_c > 3: 
                box_x = (to_c - 1) * SQUARE_SIZE
            else:
                box_x = (to_c + 1) * SQUARE_SIZE
            box_y = to_r * SQUARE_SIZE
            
            if box_y + SQUARE_SIZE * 4 > SCREEN_HEIGHT:
                box_y = SCREEN_HEIGHT - SQUARE_SIZE * 4
            
            bg_rect = pygame.Rect(box_x, box_y, SQUARE_SIZE, SQUARE_SIZE * 4)
            pygame.draw.rect(self.screen, BOARD_LIGHT_COLOR, bg_rect)
            pygame.draw.rect(self.screen, BOARD_DARK_COLOR, bg_rect, 2)
            
            self.promotion_choice_rects = {} 
            
            # --- DÜZELTME BAŞLANGICI ---
            # 4 taşı (Q, R, B, N) çiz
            # PROMOTION_PIECES_SYMBOLS listesi artık [5, 4, 3, 2] (chess.QUEEN, chess.ROOK...)
            for i, piece_type in enumerate(PROMOTION_PIECES_SYMBOLS):
                rect = pygame.Rect(box_x, box_y + i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                
                # piece_type (örn. 5) -> 'Q' (BÜYÜK harf)
                # Bu, oyuncunun Beyaz olduğunu varsayar (doğru)
                symbol = chess.piece_symbol(piece_type).upper() 
                
                # 'Q' anahtarını self.assets'te ara
                if symbol in self.assets:
                    img = self.assets[symbol] # self.assets['Q']
                    self.screen.blit(img, rect.topleft)
                
                # Tıklama için küçük harf anahtarı sakla ('q')
                self.promotion_choice_rects[symbol.lower()] = rect
            # --- DÜZELTME SONU ---

    def draw_game_over_screen(self):
        if not self.game_over:
            return
            
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill(GAMEOVER_OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))
        
        card_width, card_height = 400, 200
        card_x = (SCREEN_WIDTH - card_width) // 2
        card_y = (SCREEN_HEIGHT - card_height) // 2
        card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
        
        pygame.draw.rect(self.screen, GAMEOVER_BOX_COLOR, card_rect, border_radius=10)
        
        result = self.board.result(claim_draw=True)
        if result == '1-0': msg = "OYUN BİTTİ! Kazandınız!"
        elif result == '0-1': msg = "OYUN BİTTİ! Bot Kazandı!"
        else: msg = "OYUN BİTTİ! Berabere."
            
        title_surf = self.font_gameover_title.render(msg, True, GAMEOVER_TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(card_rect.centerx, card_rect.centery - 30))
        self.screen.blit(title_surf, title_rect)
        
        sub_surf = self.font_gameover_sub.render(
            "Yeni oyun için tıklayın veya 'N' tuşuna basın.", 
            True, (80, 80, 80)
        )
        sub_rect = sub_surf.get_rect(center=(card_rect.centerx, card_rect.centery + 40))
        self.screen.blit(sub_surf, sub_rect)

    def update_captures_and_score(self):
        current_counts = Counter(p.symbol() for p in self.board.piece_map().values())
        self.captured_by_white = []
        self.captured_by_black = []
        
        for symbol, initial_count in INITIAL_PIECE_COUNTS.items():
            current_count = current_counts.get(symbol, 0)
            diff = initial_count - current_count
            if diff > 0:
                piece = chess.Piece.from_symbol(symbol)
                for _ in range(diff):
                    if piece.color == chess.BLACK:
                        self.captured_by_white.append(piece.piece_type)
                    else:
                        self.captured_by_black.append(piece.piece_type)
        
        self.white_score = sum(PIECE_VALUES.get(pt, 0) for pt in self.captured_by_white)
        self.black_score = sum(PIECE_VALUES.get(pt, 0) for pt in self.captured_by_black)
        self.material_advantage = self.white_score - self.black_score

    def pixels_to_square(self, pos):
        x, y = pos
        if x >= BOARD_WIDTH: return None
        c = x // SQUARE_SIZE
        r = y // SQUARE_SIZE
        return chess.square(file_index=c, rank_index=7 - r)

    def handle_click(self, pos):
        if not self.human_turn or self.game_over:
            return

        clicked_square = self.pixels_to_square(pos)
        if clicked_square is None:
            return

        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = clicked_square
                self.legal_moves_for_selected = [
                    m for m in self.board.legal_moves if m.from_square == clicked_square
                ]
        else:
            from_square = self.selected_square
            to_square = clicked_square
            
            self.selected_square = None
            self.legal_moves_for_selected = []

            if from_square == to_square: return
                
            move_queen = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            
            if move_queen in self.board.legal_moves:
                self.promotion_pending = True
                self.promotion_move_coords = (from_square, to_square)
                return 
                
            move_normal = chess.Move(from_square, to_square)
            if move_normal in self.board.legal_moves:
                self.board.push(move_normal)
                self.update_captures_and_score()
                self.human_turn = False
                if not self.check_game_over():
                    self.trigger_bot_move()

    def handle_promotion_click(self, pos):
        if not self.promotion_pending:
            return
            
        for piece_char, rect in self.promotion_choice_rects.items():
            if rect.collidepoint(pos):
                from_sq, to_sq = self.promotion_move_coords
                
                if piece_char == 'q': promo_piece_type = chess.QUEEN
                elif piece_char == 'r': promo_piece_type = chess.ROOK
                elif piece_char == 'b': promo_piece_type = chess.BISHOP
                else: promo_piece_type = chess.KNIGHT
                    
                move = chess.Move(from_sq, to_sq, promotion=promo_piece_type)
                
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.update_captures_and_score()
                    self.human_turn = False
                    if not self.check_game_over():
                        self.trigger_bot_move()
                
                self.reset_promotion_state()
                return

        print("Terfi iptal edildi.")
        self.reset_promotion_state()

    def reset_promotion_state(self):
        self.promotion_pending = False
        self.promotion_move_coords = None
        self.promotion_choice_rects = {}
            
    def trigger_bot_move(self):
        threading.Thread(target=self.bot_think_thread, daemon=True).start()

    def bot_think_thread(self):
        original_sims = config.MCTS_SIMULATIONS
        config.MCTS_SIMULATIONS = GUI_MCTS_SIMULATIONS
        ai_move, _, _ = self.mcts_search.get_best_move(self.board, temperature=0.0)
        config.MCTS_SIMULATIONS = original_sims
        self.bot_move_queue.put(ai_move)

    def check_bot_queue(self):
        if self.game_over or self.human_turn or self.promotion_pending:
            return
        try:
            ai_move = self.bot_move_queue.get_nowait()
            if ai_move and ai_move in self.board.legal_moves:
                self.board.push(ai_move)
                self.update_captures_and_score()
                self.check_game_over()
                if not self.game_over:
                    self.human_turn = True
            else: self.human_turn = True
        except queue.Empty: pass

    def check_game_over(self):
        if self.board.is_game_over(claim_draw=True) and not self.game_over:
            self.game_over = True
            self.human_turn = False
            print("Oyun Bitti.")
            return True
        return False

    def start_new_game(self):
        self.board.reset()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.human_turn = True
        self.game_over = False
        self.reset_promotion_state()
        self.captured_by_white = []
        self.captured_by_black = []
        self.update_captures_and_score()
        while not self.bot_move_queue.empty():
            self.bot_move_queue.get()
        print("Yeni oyun başlatıldı.")

    def main_loop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.game_over:
                        self.start_new_game()
                    elif self.promotion_pending:
                        self.handle_promotion_click(event.pos)
                    else:
                        self.handle_click(event.pos)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        self.start_new_game()

            self.check_bot_queue()

            self.draw_board()
            self.draw_info_panel() 
            self.draw_highlights()
            self.draw_pieces()
            self.draw_promotion_choice()
            
            if self.game_over:
                self.draw_game_over_screen() 

            pygame.display.flip()
            self.clock.tick(30) # 30 FPS

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # app = ChessGUI_Pygame() çağrısı zaten __init__ içinde
    # 'assets' kontrolünü yapacaktır. Tekrar kontrol etmeye gerek yok.
    app = ChessGUI_Pygame()
    app.main_loop()