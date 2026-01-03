# DOSYA: main.py

import os
import torch
import torch.multiprocessing as mp
import numpy as np
import time
from typing import List

# rl_chess paketinden modülleri içe aktar
from rl_chess import config, network, self_play, train, evaluate

# --- Worker Function (Parallel Process) ---
def self_play_worker(worker_id: int, model_path: str, num_games: int, device_str: str):
    """
    Bağımsız bir süreç (process) olarak çalışır. 
    Modeli yükler, belirtilen sayıda oyun oynar ve verileri döndürür.
    """
    # 1. Her worker kendi model örneğini oluşturur
    # CUDA bağlamını bozmamak için worker içinde yeniden cihaz tanımlıyoruz
    device = torch.device(device_str)
    
    worker_model = network.PolicyValueNet().to(device)
    
    # Ağırlıkları yükle
    try:
        worker_model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        # Model dosyası bozuksa veya yoksa hata bas, ama süreci çökertme
        print(f"[Worker {worker_id}] Model yükleme hatası: {e}")
        return []

    worker_model.eval()
    
    all_game_data = []
    
    for i in range(num_games):
        # Oyunu oyna
        # Not: self_play.play_game fonksiyonu CPU yoğunlukludur (MCTS)
        # Ancak sinir ağı tahmini için GPU kullanır.
        game_data = self_play.play_game(worker_model)
        all_game_data.extend(game_data)
        
        # Her oyun bittiğinde bilgi ver (isteğe bağlı, çok sık olmamalı)
        if (i+1) % 1 == 0:
            print(f"[Worker {worker_id}] Oyun {i+1}/{num_games} tamamlandı.")
            
    return all_game_data

def main():
    # Multiprocessing başlatma yöntemini ayarla ('spawn' CUDA için en güvenlisidir)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("=== Satranç RL Projesi: Paralel Öğrenme Döngüsü Başlatılıyor ===")
    print(f"Cihaz: {config.DEVICE}")
    
    # İşlemci çekirdek sayısı kadar worker kullanabiliriz
    # A100 Colab'de genelde 12 core vardır.
    num_workers = min(os.cpu_count(), 12) 
    print(f"Parallel Worker Sayısı: {num_workers}")
    
    # 1. Klasörleri Oluştur
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)

    # 2. Ana Modeli Başlat
    main_model = network.PolicyValueNet().to(config.DEVICE)
    trainer = train.Trainer(main_model)
    
    # Eğer önceden eğitilmiş bir model varsa yükle
    model_path = os.path.join(config.MODEL_SAVE_PATH, config.BEST_MODEL_NAME)
    if os.path.exists(model_path):
        print(f"Mevcut en iyi model yükleniyor: {model_path}")
        trainer.load_model(config.BEST_MODEL_NAME)
    else:
        print("Sıfırdan başlanıyor.")

    # 3. Ana Döngü
    for iteration in range(1, config.NUM_ITERATIONS + 1):
        print(f"\n################################################################")
        print(f"ITERATION {iteration} / {config.NUM_ITERATIONS}")
        print(f"################################################################")
        
        iteration_start_time = time.time()
        
        # --- A. Self-Play (Paralel) ---
        print(f"\n[Aşaması 1/3] Parallel Self-Play Başlıyor ({config.NUM_SELF_PLAY_GAMES} oyun)...")
        
        # Mevcut modeli geçici olarak diske kaydet (Worker'lar okuyabilsin diye)
        temp_model_path = os.path.join(config.MODEL_SAVE_PATH, "temp_worker_model.pth")
        torch.save(main_model.state_dict(), temp_model_path)
        
        # Oyunları worker'lara dağıt
        total_games = config.NUM_SELF_PLAY_GAMES
        games_per_worker = total_games // num_workers
        remainder = total_games % num_workers
        
        worker_args = []
        for i in range(num_workers):
            # Oyunları worker'lara adil dağıt
            count = games_per_worker + (1 if i < remainder else 0)
            if count > 0:
                worker_args.append((i, temp_model_path, count, str(config.DEVICE)))
        
        new_data_count = 0
        
        # Havuz (Pool) ile çalıştır
        if len(worker_args) > 0:
            with mp.Pool(processes=len(worker_args)) as pool:
                # starmap, argümanları otomatik dağıtır
                results = pool.starmap(self_play_worker, worker_args)
                
            # Sonuçları topla
            for worker_data in results:
                trainer.add_to_buffer(worker_data)
                new_data_count += len(worker_data)
        
        # Geçici dosyayı sil
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            
        print(f"--> Toplam {new_data_count} yeni veri noktası toplandı.")
        print(f"--> Replay Buffer boyutu: {len(trainer.replay_buffer)}")
        
        # --- B. Eğitim (Training) ---
        print(f"\n[Aşaması 2/3] Eğitim...")
        
        # Eğer buffer'da yeterince veri varsa eğit
        if len(trainer.replay_buffer) >= config.BATCH_SIZE:
            for epoch in range(config.TRAIN_EPOCHS):
                avg_loss, avg_v, avg_p = trainer.train_epoch()
                print(f"  - Epoch {epoch+1}: Loss={avg_loss:.4f} (V={avg_v:.4f}, P={avg_p:.4f})")
            
            trainer.save_model(config.BEST_MODEL_NAME)
        else:
            print("Yeterli veri yok, eğitim atlandı.")
        
        # --- C. Değerlendirme ---
        print(f"\n[Aşaması 3/3] Değerlendirme...")
        # Random bot'a karşı 5 oyun
        evaluate.evaluate_vs_random(main_model, num_games=5, simulations=config.MCTS_SIMULATIONS)
        
        # Stockfish (varsa)
        if iteration % 5 == 0 and config.STOCKFISH_PATH:
             evaluate.evaluate_vs_stockfish(main_model, config.STOCKFISH_PATH, num_games=4, skill_level=config.STOCKFISH_SKILL_LEVEL)
        
        duration = time.time() - iteration_start_time
        print(f"\nIteration {iteration} tamamlandı. Geçen süre: {duration:.1f} saniye.")

if __name__ == "__main__":
    main()