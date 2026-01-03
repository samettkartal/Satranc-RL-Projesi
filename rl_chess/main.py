# DOSYA: rl_chess/main.py

import os
import torch
import torch.multiprocessing as mp
import numpy as np
import time
from typing import List
from tqdm import tqdm
from functools import partial
import sys

# rl_chess paketinden modülleri içe aktar
from rl_chess import config, network, self_play, train, evaluate

# --- Worker Function (Parallel Process) ---
def run_parallel_game(model_weights: dict, i: int) -> list:
    """
    Tek bir paralel oyun çalıştıran işçi (worker) fonksiyonu.
    """
    # 1. Her işçi KENDİ model kopyasını oluşturur
    model = network.PolicyValueNet().to(config.DEVICE)
    
    # 2. Ana modelin ağırlıklarını yükler
    model.load_state_dict(model_weights)
    model.eval()
    
    # 3. Bir oyun oyna
    try:
        game_data = self_play.play_game(model)
        return game_data
    except Exception as e:
        print(f"Hata: İşçi {i} oyun oynarken çöktü: {e}", file=sys.stderr)
        return []

def main():
    # Multiprocessing başlatma yöntemini ayarla
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"=== Satranç RL Projesi: Paralel Öğrenme Döngüsü Başlatılıyor ===")
    print(f"Cihaz: {config.DEVICE}")
    print(f"Ayarlar: {config.NUM_ITERATIONS} İterasyon, {config.NUM_SELF_PLAY_GAMES} Oyun/İterasyon")
    
    # Colab'ın bize verdiği TÜM çekirdekleri kullanalım
    # A100'de genelde 12 core vardır.
    num_workers = os.cpu_count()
    if num_workers is None: num_workers = 4
    
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
        try:
            trainer.load_model(config.BEST_MODEL_NAME)
        except:
            print("Model yüklenemedi, sıfırdan başlanıyor.")
    else:
        print("Sıfırdan başlanıyor.")

    # 3. Ana Döngü
    for iteration in range(1, config.NUM_ITERATIONS + 1):
        print(f"\n################################################################")
        print(f"ITERATION {iteration} / {config.NUM_ITERATIONS}")
        print(f"################################################################")
        
        iteration_start_time = time.time()
        
        # --- A. Self-Play (Paralel) ---
        print(f"Aşama 1: {config.NUM_SELF_PLAY_GAMES} oyun {num_workers} işçi ile oynanıyor...")
        
        iteration_game_data = []
        
        main_model.eval() 
        model_weights = main_model.state_dict()
        worker_func = partial(run_parallel_game, model_weights)

        # İşlem havuzunu Başlat
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_func, range(config.NUM_SELF_PLAY_GAMES)),
                total=config.NUM_SELF_PLAY_GAMES,
                desc="Self-Play (Paralel GPU)"
            ))

        for game_data in results:
            iteration_game_data.extend(game_data)
        
        trainer.add_to_buffer(iteration_game_data)
        print(f"--> Toplam {len(iteration_game_data)} yeni veri noktası toplandı.")
        print(f"--> Replay Buffer: {len(trainer.replay_buffer)}")
        
        if len(trainer.replay_buffer) < config.BATCH_SIZE:
             print("Yeterli veri yok, eğitim atlandı.")
             continue

        # --- B. Eğitim (Training) ---
        print(f"Aşama 2: Eğitim ({config.TRAIN_EPOCHS} epoch)...")
        main_model.train() 
        
        for _ in tqdm(range(config.TRAIN_EPOCHS), desc="Training"):
            avg_loss, avg_v, avg_p = trainer.train_epoch()
            
        print(f"  Loss={avg_loss:.4f} (V={avg_v:.4f}, P={avg_p:.4f})")
        
        # --- C. Kayıt ve Değerlendirme ---
        trainer.save_model(config.BEST_MODEL_NAME)
        
        if iteration % 5 == 0 and config.STOCKFISH_PATH:
             evaluate.evaluate_vs_stockfish(main_model, config.STOCKFISH_PATH, num_games=4, skill_level=config.STOCKFISH_SKILL_LEVEL)
        
        duration = time.time() - iteration_start_time
        print(f"İterasyon süresi: {duration:.1f} sn")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)
    main()
