import chess
import chess.engine
import torch
import os
import sys
from rl_chess import config, network, mcts

def run_benchmark():
    print("=== RL Model vs Stockfish Benchmark Başlatılıyor ===")

    # 1. Modeli Yükle
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model (1).pth")
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return

    print(f"Model yükleniyor: {model_path}")
    device = config.DEVICE
    model = network.PolicyValueNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        return

    # 2. Stockfish Kontrolü
    stockfish_path = config.STOCKFISH_PATH
    if not os.path.exists(stockfish_path):
        print(f"HATA: Stockfish bulunamadı: {stockfish_path}")
        print("Lütfen config.py dosyasındaki STOCKFISH_PATH değerini kontrol edin.")
        return

    print(f"Stockfish bulundu: {stockfish_path}")

    # 3. Benchmark Döngüsü
    # Her seviye için kaç oyun oynanacak?
    GAMES_PER_LEVEL = 4  # 2 Beyaz, 2 Siyah
    START_LEVEL = 1
    MAX_LEVEL = 20
    
    mcts_search = mcts.MCTS(model)

    log_file = "benchmark_results_v2.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== RL Model vs Stockfish Benchmark ===\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Seviye':<8} | {'Galibiyet':<10} | {'Beraberlik':<10} | {'Mağlubiyet':<10} | {'Puan':<6}\n")
        f.write("-" * 60 + "\n")
    
    print(f"Sonuçlar {log_file} dosyasına yazılacak.")

    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            for level in range(START_LEVEL, MAX_LEVEL + 1):
                # Stockfish Seviyesini Ayarla
                engine.configure({"Skill Level": level})
                
                wins, draws, losses = 0, 0, 0
                
                print(f"Seviye {level} oynanıyor...", end="", flush=True)

                # Oyunları Oyna
                for game_idx in range(GAMES_PER_LEVEL):
                    board = chess.Board()
                    # Model Rengi: Çiftler Beyaz, Tekler Siyah
                    model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
                    
                    while not board.is_game_over(claim_draw=True):
                        if board.turn == model_color:
                            # Model Hamlesi (MCTS)
                            # Sıcaklık 0.1 (düşük exploration, en iyi hamleyi oyna)
                            best_move, _, _ = mcts_search.get_best_move(board, temperature=0.1)
                            if best_move is None:
                                print(f"HATA: Model hamle üretemedi (Seviye {level}, Oyun {game_idx+1})")
                                break
                            board.push(best_move)
                        else:
                            # Stockfish Hamlesi
                            # Hızlı olması için hamle başına 0.05 - 0.1 sn verelim
                            limit = chess.engine.Limit(time=0.1)
                            result_engine = engine.play(board, limit)
                            board.push(result_engine.move)
                    
                    # Oyun Sonucu
                    result = board.result(claim_draw=True)
                    if result == '1-0':
                        if model_color == chess.WHITE: wins += 1
                        else: losses += 1
                    elif result == '0-1':
                        if model_color == chess.BLACK: wins += 1
                        else: losses += 1
                    else:
                        draws += 1
                
                print(" Tamamlandı.")

                # İstatistikler
                score = wins + (draws * 0.5)
                row_str = f"{level:<8} | {wins:<10} | {draws:<10} | {losses:<10} | {score:<6}"
                print(row_str)
                
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(row_str + "\n")

                # Eğer model hiç puan alamazsa (0 galibiyet, 0 beraberlik) testi bitirelim mi?
                # Kullanıcı "hangi seviyeye kadar yenebiliyor" dedi.
                # Hiç puan alamadığı seviye, sınırıdır.
                if score == 0:
                    msg = f"Model, Seviye {level} karşısında hiç varlık gösteremedi. Test sonlandırılıyor."
                    print(msg)
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("-" * 60 + "\n")
                        f.write(msg + "\n")
                        f.write(f"Maksimum Başarılı Seviye: {level - 1}\n")
                    break
                    
    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nBeklenmedik bir hata oluştu: {e}")

    print("=== Benchmark Tamamlandı ===")

if __name__ == "__main__":
    run_benchmark()
