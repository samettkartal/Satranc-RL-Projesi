# DOSYA: rl_chess/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ayar dosyamızdan sabitleri içe aktaralım
from rl_chess import config

class ResidualBlock(nn.Module):
    """
    Bir adet Residual Blok (Atlama Bağlantılı Blok).
    Bu, derin ağları eğitmenin anahtarıdır.
    (Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm) + Input -> ReLU
    """
    def __init__(self, num_filters: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ana yol
        identity = x # Orijinal girişi (kimliği) sakla
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Atlama bağlantısı (skip connection)
        out += identity
        return F.relu(out)

class PolicyValueNet(nn.Module):
    """
    AlphaZero mimarisine dayalı Politika ve Değer Ağı.
    
    Girdi: (N, INPUT_CHANNELS, 8, 8) boyutlu tahta tensörü
    Çıktı 1 (Politika): (N, POLICY_OUTPUT_SIZE) boyutlu hamle logitleri
    Çıktı 2 (Değer): (N, 1) boyutlu durum değeri (-1 ila +1)
    """
    def __init__(self):
        super(PolicyValueNet, self).__init__()
        
        # --- 1. Gövde (Body) ---
        # Bu ilk katman, 13 kanallı girdiyi alır ve onu 
        # ağın ana filtresi olan CONV_FILTERS'a (örn. 128) genişletir.
        self.conv_body = nn.Conv2d(
            config.INPUT_CHANNELS, config.CONV_FILTERS, kernel_size=3, padding=1, bias=False
        )
        self.bn_body = nn.BatchNorm2d(config.CONV_FILTERS)
        
        # --- 2. Kule (Tower) ---
        # Birbiri üzerine yığılmış N adet Residual Blok.
        # Bu, ağın "düşünme" katmanıdır.
        self.residual_tower = nn.ModuleList(
            [ResidualBlock(config.CONV_FILTERS) for _ in range(config.RESIDUAL_BLOCKS)]
        )
        
        # --- 3. Politika Başı (Policy Head) ---
        # Gövdeden gelen özellikleri alır ve 4096 hamle için olasılık üretir.
        self.policy_conv = nn.Conv2d(
            config.CONV_FILTERS, 2, kernel_size=1, bias=False
        ) # 128 filtreden 2 filtreye (daha basit bir temsil)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * config.BOARD_SIZE * config.BOARD_SIZE, # 2 * 8 * 8 = 128
            config.POLICY_OUTPUT_SIZE # 4096
        )
        
        # --- 4. Değer Başı (Value Head) ---
        # Gövdeden gelen özellikleri alır ve -1 ile +1 arası tek bir sayı üretir.
        self.value_conv = nn.Conv2d(
            config.CONV_FILTERS, 1, kernel_size=1, bias=False
        ) # 128 filtreden 1 filtreye (özet)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(
            1 * config.BOARD_SIZE * config.BOARD_SIZE, # 1 * 8 * 8 = 64
            256 # Ara katman
        )
        self.value_fc2 = nn.Linear(256, 1) # Tek çıktı

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Veriyi ağ üzerinden ileri besler.
        """
        # Girdiyi GPU/CPU'ya gönder
        x = x.to(config.DEVICE)
        
        # 1. Gövdeden geçir
        x = F.relu(self.bn_body(self.conv_body(x)))
        
        # 2. Residual Kulesinden geçir
        for block in self.residual_tower:
            x = block(x)
            
        # 3. Politika Başını hesapla
        pol = F.relu(self.policy_bn(self.policy_conv(x)))
        pol = torch.flatten(pol, 1) # (N, 2*8*8)
        policy_logits = self.policy_fc(pol) # (N, 4096)
        
        # 4. Değer Başını hesapla
        val = F.relu(self.value_bn(self.value_conv(x)))
        val = torch.flatten(val, 1) # (N, 1*8*8)
        val = F.relu(self.value_fc1(val))
        value = torch.tanh(self.value_fc2(val)) # Çıktıyı [-1, +1] arasına sıkıştır
        
        return policy_logits, value

# --- Test Bloğu ---
if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırıldığında testleri yap
    print(f"network.py testi çalıştırılıyor...")
    print(f"Kullanılan cihaz: {config.DEVICE}")
    
    # Modeli oluştur ve cihaza taşı
    model = PolicyValueNet().to(config.DEVICE)
    model.eval() # Test moduna al

    # Sahte bir girdi tensörü oluştur (örn. 16'lık bir batch)
    # Boyut: (BatchSize, Kanallar, Yükseklik, Genişlik)
    dummy_batch_size = 16
    dummy_input = torch.randn(
        dummy_batch_size, 
        config.INPUT_CHANNELS, 
        config.BOARD_SIZE, 
        config.BOARD_SIZE
    ).to(config.DEVICE)

    print(f"Girdi tensör boyutu: {dummy_input.shape}")

    # Modeli çalıştır
    with torch.no_grad(): # Gradyan hesaplamaya gerek yok
        policy_logits, value = model(dummy_input)

    # Çıktı boyutlarını kontrol et
    print(f"Politika (logits) çıktı boyutu: {policy_logits.shape}")
    print(f"Değer (value) çıktı boyutu: {value.shape}")

    assert policy_logits.shape == (dummy_batch_size, config.POLICY_OUTPUT_SIZE)
    assert value.shape == (dummy_batch_size, 1)
    
    # Değerin -1 ile +1 arasında olduğunu kontrol et
    assert torch.all(value >= -1) and torch.all(value <= 1)
    
    print("\nnetwork.py başarıyla oluşturuldu ve test edildi!")
    print(f"Model {config.RESIDUAL_BLOCKS} residual blok içeriyor.")