# DOSYA: rl_chess/evaluate.py

import torch
import chess
import chess.engine
import numpy as np
import random
from typing import Tuple, Optional

from rl_chess import config, network, utils, mcts

def evaluate_vs_random(model: network.PolicyValueNet, num_games: int = 10, simulations: int = 25) -> Tuple[int, int, int]:
    """
    Modeli rastgele oynayan bir bota karşı test eder.
    Döndürür: (Model Galibiyeti, Beraberlik, Model Mağlubiyeti)
    """
    print(f"\n--- Model vs Random Bot ({num_games} oyun) ---")
    
    model_wins = 0
    draws = 0
    model_losses = 0
    
    # MCTS simülasyon sayısını ayarla
    mcts_search = mcts.MCTS(model)
    
    for i in range(num_games):
        board = chess.Board()
        # Modelin hangi renk olacağını rastgele seçelim (veya sırayla)
        # 0: Model Beyaz, 1: Model Siyah
        model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        
        while not board.is_game_over(claim_draw=True):
            if board.turn == model_color:
                # Modelin sırası: MCTS kullan
                # Düşük sıcaklık (temperature near 0) ile en iyi hamleyi seç
                best_move, _, _ = mcts_search.get_best_move(board, temperature=0.1)
                
                if best_move is None:
                    break # Hata durumu veya oyun sonu
                board.push(best_move)
            else:
                # Random botun sırası
                legal_moves = list(board.legal_moves)
                random_move = random.choice(legal_moves)
                board.push(random_move)
                
        # Oyun sonucu
        result = board.result(claim_draw=True)
        print(f"Oyun {i+1}/{num_games} Bitti. Sonuç: {result} (Model Rengi: {'Beyaz' if model_color == chess.WHITE else 'Siyah'})")
        
        if result == '1-0':
            if model_color == chess.WHITE: model_wins += 1
            else: model_losses += 1
        elif result == '0-1':
            if model_color == chess.BLACK: model_wins += 1
            else: model_losses += 1
        else:
            draws += 1
            
    print(f"Sonuçlar: Galibiyet: {model_wins}, Beraberlik: {draws}, Mağlubiyet: {model_losses}")
    return model_wins, draws, model_losses

def evaluate_vs_stockfish(model: network.PolicyValueNet, stockfish_path: str, num_games: int = 10, skill_level: int = 5) -> Tuple[int, int, int]:
    """
    Modeli Stockfish motoruna karşı test eder.
    """
    print(f"\n--- Model vs Stockfish (Seviye {skill_level}) ({num_games} oyun) ---")
    
    try:
        # Stockfish motorunu başlat
        # NOT: Windows'ta .exe uzantısı önemlidir.
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        # Stockfish ayarlarını yap (Skill Level: 0-20)
        engine.configure({"Skill Level": skill_level})
        
    except FileNotFoundError:
        print(f"HATA: Stockfish şu yolda bulunamadı: {stockfish_path}")
        print("Lütfen config.py dosyasındaki STOCKFISH_PATH değerini güncelleyin.")
        return 0, 0, 0
    except Exception as e:
        print(f"Stockfish başlatılırken hata oluştu: {e}")
        return 0, 0, 0

    model_wins = 0
    draws = 0
    model_losses = 0
    mcts_search = mcts.MCTS(model)
    
    for i in range(num_games):
        board = chess.Board()
        model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        
        while not board.is_game_over(claim_draw=True):
            if board.turn == model_color:
                # Model Hamlesi
                best_move, _, _ = mcts_search.get_best_move(board, temperature=0.1)
                if best_move is None: break
                board.push(best_move)
            else:
                # Stockfish Hamlesi
                # Hamle başına 0.1 saniye süre verelim (hızlı test için)
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
        
        # Sonuç
        res = board.result(claim_draw=True)
        print(f"Oyun {i+1}/{num_games} Sonuç: {res}")
        
        if res == '1-0':
            if model_color == chess.WHITE: model_wins += 1
            else: model_losses += 1
        elif res == '0-1':
            if model_color == chess.BLACK: model_wins += 1
            else: model_losses += 1
        else:
            draws += 1
            
    engine.quit()
    print(f"Stockfish Sonuçları: Galibiyet: {model_wins}, Beraberlik: {draws}, Mağlubiyet: {model_losses}")
    return model_wins, draws, model_losses

if __name__ == "__main__":
    print("Değerlendirme Başlatılıyor...")
    
    # 1. Modeli Yükle
    model = network.PolicyValueNet().to(config.DEVICE)
    
    # Eğer eğitilmiş model varsa yükle
    try:
        # train.py içindeki Trainer sınıfını kullanmadan doğrudan yükleyelim
        import os
        model_path = os.path.join(config.MODEL_SAVE_PATH, config.BEST_MODEL_NAME)
        if os.path.exists(model_path):
            print(f"Model yükleniyor: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        else:
            print("UYARI: Eğitilmiş model bulunamadı (best_model.pth). Rastgele ağırlıklarla test ediliyor.")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")

    model.eval()
    
    # 2. Random Bot'a Karşı Test
    evaluate_vs_random(model, num_games=2, simulations=config.MCTS_SIMULATIONS)
    
    # 3. Stockfish'e Karşı Test (Eğer yol doğruysa)
    if config.STOCKFISH_PATH and os.path.exists(config.STOCKFISH_PATH):
        evaluate_vs_stockfish(model, config.STOCKFISH_PATH, num_games=2, skill_level=config.STOCKFISH_SKILL_LEVEL)
    else:
        print("\nNOT: Stockfish yolu bulunamadığı için Stockfish testi atlandı.")
        print(f"Ayarlanan Yol: {config.STOCKFISH_PATH}")
