# DOSYA: rl_chess/train.py

import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

# Proje dosyalarımızı içe aktaralım
from rl_chess import config, network, self_play

# Replay Buffer (Hafıza) için bir tip tanımı
# collections.deque, maxlen parametresi sayesinde
# eski verileri otomatik olarak siler (FIFO).
ReplayBuffer = collections.deque[self_play.TrainingDataPoint]

class Trainer:
    """
    Modeli eğitmeyi, kaydetmeyi ve yüklemeyi yöneten sınıf.
    """
    def __init__(self, model: network.PolicyValueNet):
        self.model = model.to(config.DEVICE)
        self.replay_buffer = collections.deque(maxlen=config.REPLAY_BUFFER_SIZE)
        
        # Optimizer (Adam, ağırlıkları güncellemek için)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-4 # Overfitting'i azaltmak için (L2 Reg)
        )
        
        # Kayıp Fonksiyonları (Loss Functions)
        # 1. Değer Kaybı: Modelin tahmin ettiği değer (v) ile
        #    gerçek oyun sonucu (z) arasındaki fark.
        self.value_loss_fn = nn.MSELoss() # Ortalama Karesel Hata

        # 2. Politika Kaybı: Modelin tahmini (p) ile
        #    MCTS'nin hedef politikası (pi) arasındaki fark.
        self.policy_loss_fn = nn.CrossEntropyLoss()

    def add_to_buffer(self, game_data: List[self_play.TrainingDataPoint]):
        """
        Oynanan bir oyundan gelen tüm verileri hafızaya ekler.
        """
        self.replay_buffer.extend(game_data)

    def train_epoch(self) -> tuple[float, float, float]:
        """
        Hafızadaki verilerle model üzerinde bir tam eğitim epoch'u çalıştırır.
        """
        if len(self.replay_buffer) < config.BATCH_SIZE:
            # Eğitim için yeterli veri yoksa atla
            return 0.0, 0.0, 0.0
            
        self.model.train() # Modeli 'eğitim' moduna al (BatchNorm vb. için)
        
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        
        # Hafızadaki veri sayısı / batch boyutuna göre 
        # kaç kez döngü döneceğini hesapla
        # (Bu, 'epoch' tanımımızdır)
        num_batches = len(self.replay_buffer) // config.BATCH_SIZE
        
        if num_batches == 0:
            return 0.0, 0.0, 0.0

        for _ in range(num_batches):
            # 1. Hafızadan rastgele bir batch veri çek
            batch = random.sample(self.replay_buffer, config.BATCH_SIZE)
            
            # 2. Veriyi (state, pi, value) olarak ayır
            # 'zip(*batch)' bir listedeki demetleri ayırır
            states, pis, values = zip(*batch)
            
            # 3. NumPy dizilerini PyTorch tensörlerine dönüştür
            states_tensor = torch.tensor(
                np.array(states), dtype=torch.float32
            ).to(config.DEVICE)
            
            pis_tensor = torch.tensor(
                np.array(pis), dtype=torch.float32
            ).to(config.DEVICE)
            
            values_tensor = torch.tensor(
                np.array(values), dtype=torch.float32
            ).unsqueeze(1).to(config.DEVICE) # (B,) -> (B, 1)

            # 4. Modeli çalıştır (Forward Pass)
            policy_logits, value_pred = self.model(states_tensor)
            
            # 5. Kayıpları (Loss) Hesapla
            value_loss = self.value_loss_fn(value_pred, values_tensor)
            policy_loss = self.policy_loss_fn(policy_logits, pis_tensor)
            
            # AlphaZero, bu iki kaybı basitçe toplar
            loss = value_loss + policy_loss
            
            # 6. Geri Yayılım (Backpropagation)
            self.optimizer.zero_grad() # Gradyanları sıfırla
            loss.backward()           # Yeni gradyanları hesapla
            self.optimizer.step()     # Ağırlıkları güncelle

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            
        # Ortalama kayıpları döndür
        avg_loss = total_loss / num_batches
        avg_v_loss = total_value_loss / num_batches
        avg_p_loss = total_policy_loss / num_batches

        # Log dosyasına kaydet
        self.log_metrics(avg_loss, avg_v_loss, avg_p_loss)
        
        return avg_loss, avg_v_loss, avg_p_loss

    def log_metrics(self, avg_loss: float, value_loss: float, policy_loss: float):
        """
        Eğitim metriklerini CSV dosyasına ekler.
        """
        file_exists = os.path.exists(config.TRAIN_LOG_PATH)
        
        with open(config.TRAIN_LOG_PATH, 'a') as f:
            if not file_exists:
                # Başlıkları yaz
                f.write("avg_loss,value_loss,policy_loss\n")
            
            # Veriyi yaz
            f.write(f"{avg_loss:.5f},{value_loss:.5f},{policy_loss:.5f}\n")


    def save_model(self, filename: str):
        """
        Modelin mevcut ağırlıklarını bir dosyaya kaydeder.
        """
        if not os.path.exists(config.MODEL_SAVE_PATH):
            os.makedirs(config.MODEL_SAVE_PATH)
            
        filepath = os.path.join(config.MODEL_SAVE_PATH, filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model şuraya kaydedildi: {filepath}")

    def load_model(self, filename: str):
        """
        Kaydedilmiş model ağırlıklarını yükler.
        """
        filepath = os.path.join(config.MODEL_SAVE_PATH, filename)
        if not os.path.exists(filepath):
            print(f"Uyarı: Model dosyası bulunamadı: {filepath}. "
                  "Model rastgele ağırlıklarla başlıyor.")
            return
            
        # map_location=config.DEVICE: Modeli GPU/CPU'ya uygun şekilde yükler
        self.model.load_state_dict(
            torch.load(filepath, map_location=config.DEVICE)
        )
        print(f"Model şuradan yüklendi: {filepath}")

# --- Test Bloğu ---
if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırıldığında testleri yap
    print("train.py testi çalıştırılıyor...")

    # 1. Sahte model ve eğitmen oluştur
    test_model = network.PolicyValueNet().to(config.DEVICE)
    trainer = Trainer(test_model)

    # 2. Sahte eğitim verisi (TrainingDataPoint) oluştur
    # (self_play.py'yi çalıştırmak yerine simüle ediyoruz)
    fake_game_data = []
    num_fake_data = config.BATCH_SIZE * 2 # 2 batch'lik veri
    
    for _ in range(num_fake_data):
        fake_state = np.random.rand(
            config.INPUT_CHANNELS, 8, 8
        ).astype(np.float32)
        
        fake_pi = np.random.rand(
            config.POLICY_OUTPUT_SIZE
        ).astype(np.float32)
        fake_pi = fake_pi / np.sum(fake_pi) # Normalleştir
        
        fake_value = random.choice([-1.0, 0.0, 1.0])
        
        fake_game_data.append((fake_state, fake_pi, fake_value))

    # 3. Veriyi hafızaya ekle
    trainer.add_to_buffer(fake_game_data)
    print(f"Hafızaya {len(trainer.replay_buffer)} adet sahte veri eklendi.")
    assert len(trainer.replay_buffer) == num_fake_data

    # 4. Bir epoch eğit
    print("Test eğitim epoch'u başlatılıyor...")
    avg_loss, avg_v, avg_p = trainer.train_epoch()

    print(f"Eğitim tamamlandı.")
    print(f"Ortalama Toplam Kayıp (Loss): {avg_loss:.4f}")
    print(f"Ortalama Değer Kaybı (V_Loss): {avg_v:.4f}")
    print(f"Ortalama Politika Kaybı (P_Loss): {avg_p:.4f}")
    
    # Kaybın hesaplanıp hesaplanmadığını kontrol et
    assert avg_loss > 0
    
    # 5. Model kaydetme/yükleme testi
    test_filename = "test_model_weights.pth"
    trainer.save_model(test_filename)
    
    # Yeni bir model oluştur ve yükle
    new_model = network.PolicyValueNet().to(config.DEVICE)
    new_trainer = Trainer(new_model)
    new_trainer.load_model(test_filename)
    
    print("Model kaydetme ve yükleme testi başarılı.")
    
    # Test dosyasını temizle
    os.remove(os.path.join(config.MODEL_SAVE_PATH, test_filename))
    
    print("\ntrain.py başarıyla oluşturuldu ve test edildi!")