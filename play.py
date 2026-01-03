# DOSYA: play.py

import torch
import chess
import chess.svg
import time
import os

from rl_chess import config, network, mcts, utils

def print_board(board: chess.Board, last_move: chess.Move = None):
    """
    Tahtayı konsola basar. 
    (Eğer bir GUI'de değilseniz, bu en basit görselleştirme yoludur).
    """
    os.system('cls' if os.name == 'nt' else 'clear') # Konsolu temizle
    
    # Tahtayı Unicode karakterlerle yazdır
    print("\n--- Yapay Zeka Satranç Botu ---")
    print(board.unicode(invert_color=False, borders=True))
    
    if last_move:
        print(f"\nSon Hamle: {last_move.uci()}")
    print(f"Hamle Sırası: {'Beyaz (İnsan)' if board.turn == chess.WHITE else 'Siyah (AI Bot)'}")
    print("---------------------------------")


def get_human_move(board: chess.Board) -> chess.Move:
    """
    İnsandan (kullanıcıdan) yasal bir hamle alır.
    Hamleyi "e2e4", "g1f3", "e7e8q" (terfi) gibi UCI formatında alır.
    """
    while True:
        try:
            uci_move = input("Hamlenizi girin (örn: e2e4): ")
            move = chess.Move.from_uci(uci_move)
            
            # Piyon terfisi için otomatik vezir (q) ekle
            # Eğer kullanıcı 'e7e8' girerse ve bu bir terfi ise, 
            # python-chess bunu 'e7e8q' olarak ister.
            if move.promotion is None and \
               board.piece_at(move.from_square).piece_type == chess.PAWN and \
               (chess.square_rank(move.to_square) == 0 or \
                chess.square_rank(move.to_square) == 7):
                
                move.promotion = chess.QUEEN # Otomatik vezire terfi
            
            if move in board.legal_moves:
                return move
            else:
                print("Hata: Bu yasal bir hamle değil! Tekrar deneyin.")
        except ValueError:
            print("Hata: Geçersiz hamle formatı! UCI formatında girin (örn: e2f4).")
        except Exception as e:
            print(f"Beklenmedik bir hata oluştu: {e}")

def play_vs_ai(model: network.PolicyValueNet, 
               human_color: chess.Color = chess.WHITE):
    """
    Ana oyun döngüsü.
    """
    board = chess.Board()
    model.eval() # Modeli 'değerlendirme' moduna al
    
    # AI Bot için MCTS'yi başlat
    ai_mcts = mcts.MCTS(model)
    
    # MCTS simülasyon sayısını config'den daha yüksek ayarlayabiliriz
    # (Eğitimden daha güçlü oynaması için)
    play_mcts_simulations = 400 # config.MCTS_SIMULATIONS yerine
    
    print(f"Oyun başlıyor. Siz {'Beyaz' if human_color == chess.WHITE else 'Siyah'}sınız.")
    print(f"AI Bot her hamle için {play_mcts_simulations} MCTS simülasyonu kullanacak.")
    
    last_move_played = None

    while not board.is_game_over(claim_draw=True):
        print_board(board, last_move_played)
        
        if board.turn == human_color:
            # --- İnsan Oyuncunun Sırası ---
            move = get_human_move(board)
            board.push(move)
            last_move_played = move
        else:
            # --- AI Bot'un Sırası ---
            print("AI Bot düşünüyor...")
            
            # AI oynarken 'deterministic' (sıcaklık=0) olmalı
            # Not: mcts.py'deki get_best_move'un simülasyon sayısını 
            # dinamik alacak şekilde güncellenmesi gerekebilir, 
            # şimdilik config'deki değeri kullanacak.
            # (Basitlik için config.MCTS_SIMULATIONS'u 400 yapabilirsiniz)
            
            # Orijinal MCTS'miz config'den okuyordu, bu yüzden 
            # 'play' için config'i geçici olarak değiştirelim:
            original_sims = config.MCTS_SIMULATIONS
            config.MCTS_SIMULATIONS = play_mcts_simulations
            
            start_time = time.time()
            ai_move, _, _ = ai_mcts.get_best_move(board, temperature=0.0)
            end_time = time.time()

            # Config'i geri yükle
            config.MCTS_SIMULATIONS = original_sims

            if ai_move:
                print(f"AI Bot'un hamlesi: {ai_move.uci()} "
                      f"({end_time - start_time:.2f} saniyede düşünüldü)")
                board.push(ai_move)
                last_move_played = ai_move
            else:
                print("AI Bot hamle bulamadı (Oyun sonu mu?).")
                break
        
        time.sleep(0.5) # Tahtayı görmek için kısa bir bekleme

    # --- Oyun Bitti ---
    print_board(board, last_move_played)
    print("\n--- OYUN BİTTİ! ---")
    result = board.result(claim_draw=True)
    if result == '1-0':
        print("Beyaz kazandı!")
    elif result == '0-1':
        print("Siyah kazandı!")
    elif result == '1/2-1/2':
        print("Oyun berabere bitti!")
    else:
        print(f"Oyun sonucu: {result}")

if __name__ == "__main__":
    # --- Modeli Yükle ---
    print(f"AI Bot'un 'beyni' ({config.BEST_MODEL_NAME}) yükleniyor...")
    
    if not os.path.exists(os.path.join(config.MODEL_SAVE_PATH, config.BEST_MODEL_NAME)):
        print(f"HATA: Model dosyası bulunamadı: {config.BEST_MODEL_NAME}")
        print("Lütfen önce 'main.py' betiğini çalıştırarak bir model eğitin.")
    else:
        # Modeli oluştur
        model = network.PolicyValueNet().to(config.DEVICE)
        
        # Eğitilmiş ağırlıkları yükle
        model.load_state_dict(
            torch.load(
                os.path.join(config.MODEL_SAVE_PATH, config.BEST_MODEL_NAME), 
                map_location=config.DEVICE
            )
        )
        print("Model başarıyla yüklendi. Oyun başlatılıyor...")
        
        # Oyunu başlat (İnsan = Beyaz)
        play_vs_ai(model, human_color=chess.WHITE)