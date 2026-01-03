# DOSYA: gui_stockfish.py (Sürüm 2: Renk Seçim Menüsü Eklendi)
# Açıklama: Stockfish motorunu kullanan ve oyun başında
# Beyaz/Siyah seçimi sunan profesyonel Pygame arayüzü.

import pygame
import chess
import threading
import queue
import os
import sys
from collections import Counter
from stockfish import Stockfish

# --- Arayüz Ayarları ---
BOARD_WIDTH = 640
INFO_PANEL_WIDTH = 200
SCREEN_WIDTH = BOARD_WIDTH + INFO_PANEL_WIDTH
SCREEN_HEIGHT = BOARD_WIDTH
SQUARE_SIZE = BOARD_WIDTH // 8
MINI_SQUARE_SIZE = INFO_PANEL_WIDTH // 6 

# Renkler
BOARD_LIGHT_COLOR = (234, 221, 197)
BOARD_DARK_COLOR = (168, 136, 101)
INFO_PANEL_BG = (40, 40, 40)
SELECTED_COLOR = (247, 196, 77, 150)
LEGAL_MOVE_COLOR = (20, 80, 20, 100)
TEXT_COLOR = (255, 255, 255)
GAMEOVER_OVERLAY_COLOR = (0, 0, 0, 180)
GAMEOVER_BOX_COLOR = (230, 230, 230)
GAMEOVER_TEXT_COLOR = (10, 10, 10)
MENU_BG_COLOR = (30, 30, 30)
MENU_BUTTON_COLOR = (80, 80, 80)
MENU_BUTTON_HOVER_COLOR = (110, 110, 110)

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
PROMOTION_PIECES_SYMBOLS = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
CAPTURES_BG_FILENAME = "captures_bg.png"

# --- Stockfish Ayarları ---
# Lütfen .exe dosyanızın adını buraya tam olarak yazın
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"
STOCKFISH_DIFFICULTY = 20 # 0 (en kolay) ile 20 (en zor) arası


class ChessGUI_Pygame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Stockfish vs İnsan")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fontlar
        self.font_large = pygame.font.SysFont("Arial", 30, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 20)
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_gameover_title = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_gameover_sub = pygame.font.SysFont("Arial", 18)
        self.font_menu_title = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_menu_button = pygame.font.SysFont("Arial", 24)

        # --- 1. Modeli ve Görselleri Yükle ---
        self.stockfish_engine = self.load_stockfish_engine()
        if self.stockfish_engine is None:
            print(f"HATA: Stockfish motoru yüklenemedi. '{STOCKFISH_PATH}' yolunun doğru olduğundan emin olun.")
            sys.exit(1)
            
        self.assets = self.load_assets()
        if not self.assets.get('P'):
            print("\n!!! YÜKLEME BAŞARISIZ !!!")
            print("HATA: 'assets' klasöründeki taş resimleri yüklenemedi.")
            sys.exit(1)
        
        # --- 2. Oyun Durumu Değişkenleri ---
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves_for_selected = []
        
        # --- YENİ: Oyun Durumu ve Renk Yönetimi ---
        self.game_state = 'MENU' # 'MENU', 'PLAYING'
        self.human_color = None  # chess.WHITE veya chess.BLACK
        self.human_turn = False
        self.game_over = False
        
        # Menü Butonları
        self.play_white_button_rect = pygame.Rect((SCREEN_WIDTH / 2) - 150, 250, 300, 60)
        self.play_black_button_rect = pygame.Rect((SCREEN_WIDTH / 2) - 150, 350, 300, 60)

        # Puanlama
        self.captured_by_white = [] 
        self.captured_by_black = [] 
        self.material_advantage = 0 
        
        # Terfi
        self.promotion_pending = False
        self.promotion_move_coords = None
        self.promotion_choice_rects = {}
        
        # --- 3. Thread Yönetimi ---
        self.bot_move_queue = queue.Queue()

    #
    # --- load_stockfish_engine ve load_assets DEĞİŞMEDİ ---
    #
    def load_stockfish_engine(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(script_dir, STOCKFISH_PATH)
        
        if not os.path.exists(engine_path): 
            print(f"HATA: Stockfish motoru beklenen yolda bulunamadı: {engine_path}")
            print("Lütfen Stockfish'i indirip STOCKFISH_PATH değişkenini güncelleyin.")
            return None
        try:
            stockfish = Stockfish(path=engine_path, parameters={
                "Skill Level": STOCKFISH_DIFFICULTY
            })
            print(f"Stockfish motoru başarıyla yüklendi: {engine_path}")
            print(f"Stockfish zorluk seviyesi: {STOCKFISH_DIFFICULTY}")
            return stockfish
        except Exception as e:
            print(f"HATA: Stockfish yüklenemedi! {e}")
            return None

    def load_assets(self):
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
                img = pygame.image.load(path)
                if img is None: raise Exception("Pygame resmi yükleyemedi (None döndürdü).")
                img = img.convert_alpha()
                if img is None: raise Exception("Resim 'convert_alpha()' yapılamadı.")
                images[symbol] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
                images[f"{symbol}_mini"] = pygame.transform.smoothscale(img, (MINI_SQUARE_SIZE, MINI_SQUARE_SIZE))
            except Exception as e: 
                print("="*50)
                print(f"!!! HATA (Durum 2): BİR DOSYA YÜKLENEMEDİ !!!")
                print(f"Hata yaşanan dosya: {path}")
                print(f"Hata mesajı: {e}")
                print("Lütfen bu PNG dosyasını silip 'assets' klasörüne yeniden indirin.")
                print("="*50)
                return {}
        bg_path = os.path.join(assets_path, CAPTURES_BG_FILENAME)
        if os.path.exists(bg_path):
            try:
                images['captures_bg'] = pygame.image.load(bg_path).convert_alpha()
                print(f"'{CAPTURES_BG_FILENAME}' arka planı başarıyla yüklendi.")
            except Exception as e:
                print(f"Uyarı: '{CAPTURES_BG_FILENAME}' bozuk. Hata: {e}")
                images['captures_bg'] = None
        else:
            images['captures_bg'] = None
            print(f"Uyarı: '{CAPTURES_BG_FILENAME}' bulunamadı. Düz renk kullanılacak.")
        print("Görseller (büyük ve minik) başarıyla yüklendi.")
        return images

    #
    # --- Çizim Fonksiyonları ('draw_...' ile başlayanlar) DEĞİŞMEDİ ---
    #
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
        adv_text = "0"
        if self.material_advantage > 0: adv_text = f"+{self.material_advantage}"
        elif self.material_advantage < 0: adv_text = f"{self.material_advantage}"
        adv_surf = self.font_large.render(adv_text, True, TEXT_COLOR)
        adv_rect = adv_surf.get_rect(center=(BOARD_WIDTH + INFO_PANEL_WIDTH // 2, 40))
        self.screen.blit(adv_surf, adv_rect)
        white_cap_bg_rect = pygame.Rect(BOARD_WIDTH + 10, 105, INFO_PANEL_WIDTH - 20, 200)
        black_cap_bg_rect = pygame.Rect(BOARD_WIDTH + 10, 345, INFO_PANEL_WIDTH - 20, 200)
        if self.assets.get('captures_bg'):
            try:
                scaled_bg = pygame.transform.scale(self.assets['captures_bg'], white_cap_bg_rect.size)
                self.screen.blit(scaled_bg, white_cap_bg_rect)
                self.screen.blit(scaled_bg, black_cap_bg_rect)
            except Exception:
                pygame.draw.rect(self.screen, (30,30,30), white_cap_bg_rect)
                pygame.draw.rect(self.screen, (30,30,30), black_cap_bg_rect)
        else:
            pygame.draw.rect(self.screen, (30,30,30), white_cap_bg_rect)
            pygame.draw.rect(self.screen, (30,30,30), black_cap_bg_rect)
        y_offset = 80
        x_offset = BOARD_WIDTH + 10
        bot_captures_surf = self.font_small.render("Botun Aldıkları:", True, (180,180,180))
        self.screen.blit(bot_captures_surf, (x_offset, y_offset))
        y_offset = 110 
        sorted_captures = sorted(self.captured_by_black, key=lambda p: PIECE_VALUES.get(p, 0), reverse=True)
        for piece_type in sorted_captures:
            symbol = chess.piece_symbol(piece_type).upper()
            if f"{symbol}_mini" in self.assets:
                img = self.assets[f"{symbol}_mini"]
                self.screen.blit(img, (x_offset, y_offset))
                x_offset += MINI_SQUARE_SIZE
                if x_offset > SCREEN_WIDTH - MINI_SQUARE_SIZE - 10: 
                    x_offset = BOARD_WIDTH + 10
                    y_offset += MINI_SQUARE_SIZE
        y_offset = 320 
        x_offset = BOARD_WIDTH + 10
        human_captures_surf = self.font_small.render("Sizin Aldıklarınız:", True, (180,180,180))
        self.screen.blit(human_captures_surf, (x_offset, y_offset))
        y_offset = 350
        sorted_captures = sorted(self.captured_by_white, key=lambda p: PIECE_VALUES.get(p, 0), reverse=True)
        for piece_type in sorted_captures:
            symbol = chess.piece_symbol(piece_type).lower()
            if f"{symbol}_mini" in self.assets:
                img = self.assets[f"{symbol}_mini"]
                self.screen.blit(img, (x_offset, y_offset))
                x_offset += MINI_SQUARE_SIZE
                if x_offset > SCREEN_WIDTH - MINI_SQUARE_SIZE - 10:
                    x_offset = BOARD_WIDTH + 10
                    y_offset += MINI_SQUARE_SIZE

    def draw_promotion_choice(self):
        if not self.promotion_pending: return
        from_sq, to_sq = self.promotion_move_coords
        to_c = chess.square_file(to_sq)
        to_r = 7 - chess.square_rank(to_sq)
        if to_c > 3: box_x = (to_c - 1) * SQUARE_SIZE
        else: box_x = (to_c + 1) * SQUARE_SIZE
        box_y = to_r * SQUARE_SIZE
        if box_y + SQUARE_SIZE * 4 > SCREEN_HEIGHT:
            box_y = SCREEN_HEIGHT - SQUARE_SIZE * 4
        bg_rect = pygame.Rect(box_x, box_y, SQUARE_SIZE, SQUARE_SIZE * 4)
        pygame.draw.rect(self.screen, BOARD_LIGHT_COLOR, bg_rect)
        pygame.draw.rect(self.screen, BOARD_DARK_COLOR, bg_rect, 2)
        self.promotion_choice_rects = {} 
        for i, piece_type in enumerate(PROMOTION_PIECES_SYMBOLS):
            rect = pygame.Rect(box_x, box_y + i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            # --- DEĞİŞİKLİK: İnsan Siyahsa, Siyah terfi taşlarını göster ---
            if self.human_color == chess.WHITE:
                symbol = chess.piece_symbol(piece_type).upper()
            else:
                symbol = chess.piece_symbol(piece_type).lower()
                
            if symbol in self.assets:
                img = self.assets[symbol]
                self.screen.blit(img, rect.topleft)
            self.promotion_choice_rects[chess.piece_symbol(piece_type).lower()] = rect

    def draw_game_over_screen(self):
        if not self.game_over: return
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill(GAMEOVER_OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))
        card_width, card_height = 400, 200
        card_x = (SCREEN_WIDTH - card_width) // 2
        card_y = (SCREEN_HEIGHT - card_height) // 2
        card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
        pygame.draw.rect(self.screen, GAMEOVER_BOX_COLOR, card_rect, border_radius=10)
        result = self.board.result(claim_draw=True)
        # --- DEĞİŞİKLİK: Kazanma mesajı seçilen renge göre ---
        if result == '1-0': 
            msg = "OYUN BİTTİ! Beyaz Kazandı!"
        elif result == '0-1': 
            msg = "OYUN BİTTİ! Siyah Kazandı!"
        else: 
            msg = "OYUN BİTTİ! Berabere."
        title_surf = self.font_gameover_title.render(msg, True, GAMEOVER_TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(card_rect.centerx, card_rect.centery - 30))
        self.screen.blit(title_surf, title_rect)
        sub_surf = self.font_gameover_sub.render(
            "Yeni oyun için tıklayın veya 'N' tuşuna basın.", True, (80, 80, 80)
        )
        sub_rect = sub_surf.get_rect(center=(card_rect.centerx, card_rect.centery + 40))
        self.screen.blit(sub_surf, sub_rect)

    # --- YENİ FONKSİYON: Başlangıç Menüsünü Çiz ---
    def draw_menu(self):
        self.screen.fill(MENU_BG_COLOR)
        
        # Başlık
        title_surf = self.font_menu_title.render("Stockfish'e Karşı Oyna", True, TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH / 2, 150))
        self.screen.blit(title_surf, title_rect)

        # Fare konumunu al
        mouse_pos = pygame.mouse.get_pos()

        # Beyaz Oyna Butonu
        btn_white_color = MENU_BUTTON_COLOR
        if self.play_white_button_rect.collidepoint(mouse_pos):
            btn_white_color = MENU_BUTTON_HOVER_COLOR
        pygame.draw.rect(self.screen, btn_white_color, self.play_white_button_rect, border_radius=10)
        white_text_surf = self.font_menu_button.render("Beyaz Olarak Oyna", True, TEXT_COLOR)
        white_text_rect = white_text_surf.get_rect(center=self.play_white_button_rect.center)
        self.screen.blit(white_text_surf, white_text_rect)

        # Siyah Oyna Butonu
        btn_black_color = MENU_BUTTON_COLOR
        if self.play_black_button_rect.collidepoint(mouse_pos):
            btn_black_color = MENU_BUTTON_HOVER_COLOR
        pygame.draw.rect(self.screen, btn_black_color, self.play_black_button_rect, border_radius=10)
        black_text_surf = self.font_menu_button.render("Siyah Olarak Oyna", True, TEXT_COLOR)
        black_text_rect = black_text_surf.get_rect(center=self.play_black_button_rect.center)
        self.screen.blit(black_text_surf, black_text_rect)
    
    #
    # --- Buradan sonraki tüm fonksiyonlar (update_scores, clicks vb.) ---
    # --- Yeni mantığa (self.human_color) göre güncellendi ---
    #
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
        """Tıklama mantığını yönetir (Oyun Sırasında)."""
        if not self.human_turn or self.game_over: return
        clicked_square = self.pixels_to_square(pos)
        if clicked_square is None: return
        if self.selected_square is None:
            # --- DEĞİŞİKLİK: Sadece 'self.human_color' taşlarını seç ---
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.human_color:
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
            
            # --- DEĞİŞİKLİK: Terfi hamlesini 'self.human_color'a göre kontrol et ---
            promo_piece = chess.QUEEN if self.human_color == chess.WHITE else chess.QUEEN
            move_promo = chess.Move(from_square, to_square, promotion=promo_piece)
            
            if move_promo in self.board.legal_moves:
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

    def handle_click_menu(self, pos):
        """Tıklama mantığını yönetir (Menüdeyken)."""
        if self.play_white_button_rect.collidepoint(pos):
            self.start_new_game(chess.WHITE)
        elif self.play_black_button_rect.collidepoint(pos):
            self.start_new_game(chess.BLACK)

    def handle_promotion_click(self, pos):
        if not self.promotion_pending: return
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
        try:
            self.stockfish_engine.set_fen_position(self.board.fen())
            # Düşünme süresi (zorluğu zaten belirledik)
            # 1000ms = 1 saniye. Güçlü bir bot için yeterli.
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
        if self.game_over or self.human_turn or self.promotion_pending:
            return
        try:
            ai_move = self.bot_move_queue.get_nowait()
            if ai_move and ai_move in self.board.legal_moves:
                self.board.push(ai_move)
                self.update_captures_and_score()
                self.check_game_over()
                if not self.game_over:
                    self.human_turn = True # Sırayı insana ver
            else: 
                if ai_move is not None:
                    print(f"HATA: Stockfish yasal olmayan hamle döndürdü: {ai_move.uci()}")
                self.human_turn = True
        except queue.Empty: pass

    def check_game_over(self):
        if self.board.is_game_over(claim_draw=True) and not self.game_over:
            self.game_over = True
            self.human_turn = False
            print("Oyun Bitti.")
            return True
        return False

    def start_new_game(self, human_color: chess.Color):
        """Oyunu seçilen renge göre başlatır."""
        self.board.reset()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.game_over = False
        self.reset_promotion_state()
        self.captured_by_white = []
        self.captured_by_black = []
        self.update_captures_and_score()
        while not self.bot_move_queue.empty():
            self.bot_move_queue.get()
            
        self.stockfish_engine.set_fen_position(chess.STARTING_FEN)
        
        self.human_color = human_color
        self.game_state = 'PLAYING'
        
        # --- DEĞİŞİKLİK: Siyah seçildiyse bot başlar ---
        if self.human_color == chess.WHITE:
            self.human_turn = True
            print("Yeni oyun başlatıldı (İnsan Beyaz).")
        else:
            self.human_turn = False
            print("Yeni oyun başlatıldı (İnsan Siyah). Botun hamlesi bekleniyor...")
            self.trigger_bot_move() # Bot (Beyaz) ilk hamleyi yapsın
            
    def reset_to_menu(self):
        """Oyunu sıfırlar ve menüye döner."""
        self.game_state = 'MENU'
        self.human_color = None
        self.game_over = False
        # Stockfish'i de sıfırla
        self.stockfish_engine.set_fen_position(chess.STARTING_FEN)


    def main_loop(self):
        """Ana Pygame döngüsü."""
        running = True
        while running:
            # --- Olay (Event) Yönetimi ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # --- DEĞİŞİKLİK: Tıklama mantığı oyun durumuna göre ---
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.game_state == 'MENU':
                        self.handle_click_menu(event.pos)
                    elif self.game_state == 'PLAYING':
                        if self.game_over:
                            self.reset_to_menu() # Oyun bittiyse tıkla -> Menüye dön
                        elif self.promotion_pending:
                            self.handle_promotion_click(event.pos)
                        else:
                            self.handle_click(event.pos)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        self.reset_to_menu() # 'N' tuşu -> Menüye dön

            # --- Botun Hamlesini Kontrol Et (düşünüyorsa) ---
            if self.game_state == 'PLAYING':
                self.check_bot_queue()

            # --- Çizim (Drawing) ---
            if self.game_state == 'MENU':
                self.draw_menu()
            
            elif self.game_state == 'PLAYING':
                self.draw_board()
                self.draw_info_panel() 
                self.draw_highlights()
                self.draw_pieces()
                self.draw_promotion_choice()
                if self.game_over:
                    self.draw_game_over_screen() 

            # --- Ekranı Güncelle ---
            pygame.display.flip()
            self.clock.tick(30) # 30 FPS

        pygame.quit()
        sys.exit()

# --- Uygulamayı Başlatma ---
if __name__ == "__main__":
    app = ChessGUI_Pygame()
    app.main_loop()