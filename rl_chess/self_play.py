# DOSYA: rl_chess/self_play.py

import torch
import chess
import numpy as np
from typing import List, Tuple

from rl_chess import config, utils, network, mcts

# Eğitim verisi için bir veri yapısı tanımlayalım
# (tahta_tensörü, mcts_politika_hedefi_pi, oyun_sonucu_değeri)
TrainingDataPoint = Tuple[np.ndarray, np.ndarray, float]

def play_game(model: network.PolicyValueNet) -> List[TrainingDataPoint]:
    """
    Modeli kullanarak kendi kendine tek bir oyun oynar.
    Oyun bittiğinde, eğitim için (durum, politika, değer) listesini döndürür.
    """
    
    # MCTS arama ağacını modelle başlat
    mcts_search = mcts.MCTS(model)
    
    # Oyun tahtasını başlat
    board = chess.Board()
    
    # Bu oyunda toplanan geçici veriler
    # (state_tensor, pi_target, player_turn)
    game_history: List[Tuple[np.ndarray, np.ndarray, chess.Color]] = []
    
    move_count = 0
    
    while not board.is_game_over(claim_draw=True):
        
        # --- 1. Sıcaklığı (Temperature) Belirle ---
        # Oyunun başında (config.EXPLORE_TEMPERATURE_MOVES'a kadar)
        # keşif (exploration) için daha yüksek sıcaklık kullan.
        # Bu, botun farklı açılışları denemesini sağlar.
        if move_count < config.EXPLORE_TEMPERATURE_MOVES:
            temperature = 1.0
        else:
            # Oyunun geri kalanında en iyi hamleyi seç (exploitation).
            temperature = 0.0 # (MCTS'de 0 ise argmax olarak yorumlanır)

        # --- 2. MCTS ile Hamle Seç ---
        # board_to_tensor, utils'de zaten mevcut oyuncunun 
        # perspektifine göre ayarlanmıştı.
        state_tensor = utils.board_to_tensor(board)
        
        # MCTS'yi çalıştır ve (en iyi hamle, politika_hedefi_pi) al
        best_move, policy_target_pi, _ = mcts_search.get_best_move(
            board, temperature
        )
        
        if best_move is None:
            # Pat veya mat durumu, hamle yok
            break
            
        # --- 3. Veriyi Kaydet ---
        # Tahta tensörünü, MCTS'nin "düşünülmüş" politikasını (pi)
        # ve hamle sırasını (kimin oynadığını) kaydet.
        # Değer (value) henüz bilinmiyor, oyun sonunda eklenecek.
        game_history.append((state_tensor, policy_target_pi, board.turn))
        
        # --- 4. Hamleyi Oyna ---
        board.push(best_move)
        move_count += 1

    # --- 5. Oyun Bitti: Değeri (Value) Belirle ---
    
    final_result = board.result(claim_draw=True)
    
    if final_result == '1-0':
        # Beyaz kazandı
        game_value = 1.0
    elif final_result == '0-1':
        # Siyah kazandı
        game_value = -1.0
    else:
        # Beraberlik (1/2-1/2, *, vb.)
        game_value = 0.0

    # --- 6. Eğitim Verisini Oluştur ---
    # Toplanan veriyi (game_history) dolaş ve her adıma doğru
    # değeri (game_value) ata.
    
    training_data: List[TrainingDataPoint] = []
    
    for (state_tensor, policy_target_pi, player_turn) in game_history:
        
        # Değer, *her zaman* o anki oyuncunun (player_turn)
        # bakış açısına göre olmalıdır.
        
        if player_turn == chess.WHITE:
            # Eğer Beyazın sırasındaysa ve Beyaz kazandıysa (+1), değer +1'dir.
            # Eğer Beyazın sırasındaysa ve Siyah kazandıysa (-1), değer -1'dir.
            value_for_player = game_value 
        else: # player_turn == chess.BLACK
            # Eğer Siyahın sırasındaysa ve Siyah kazandıysa (-1), Siyahın değeri +1'dir.
            # Eğer Siyahın sırasındaysa ve Beyaz kazandıysa (+1), Siyahın değeri -1'dir.
            value_for_player = -game_value
            
        training_data.append((state_tensor, policy_target_pi, value_for_player))
        
    return training_data

# --- Test Bloğu ---
if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırıldığında testleri yap
    print("self_play.py testi çalıştırılıyor...")
    print(f"Kullanılan cihaz: {config.DEVICE}")

    # Test için küçük bir model oluşturalım
    # Not: Gerçek eğitim için 'best_model.pth' yüklenir
    test_model = network.PolicyValueNet().to(config.DEVICE)
    test_model.eval()

    print("Kendi kendine test oyunu başlatılıyor...")
    print(f"MCTS Simülasyonları (test için düşük): 5")
    # Testin hızlı bitmesi için simülasyon sayısını geçici olarak düşürelim
    original_sims = config.MCTS_SIMULATIONS
    config.MCTS_SIMULATIONS = 5 
    
    training_data = play_game(test_model)
    
    # Ayarları geri yükle
    config.MCTS_SIMULATIONS = original_sims

    print(f"\nOyun bitti. Toplam {len(training_data)} hamlelik veri toplandı.")
    
    assert len(training_data) > 0 # Oyun oynanmış olmalı
    
    # Toplanan verinin formatını kontrol et
    state, pi, value = training_data[0]
    print(f"İlk veri noktası (State) boyutu: {state.shape}")
    print(f"İlk veri noktası (Pi) boyutu: {pi.shape}")
    print(f"İlk veri noktası (Value) değeri: {value}")
    
    assert state.shape == (config.INPUT_CHANNELS, 8, 8)
    assert pi.shape == (config.POLICY_OUTPUT_SIZE,)
    assert value in [1.0, -1.0, 0.0]

    print("\nself_play.py başarıyla oluşturuldu ve test edildi!")