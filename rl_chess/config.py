# DOSYA: rl_chess/config.py

import torch

# --- Proje Genel Ayarları ---
# En iyi eğitim için 'cuda' (NVIDIA GPU) şarttır.
# Eğer GPU yoksa 'cpu' olarak değiştirin, ancak çok yavaş olacaktır.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tahta ve Hamle Temsili Ayarları (State & Action Representation) ---
BOARD_SIZE = 8
INPUT_CHANNELS = 13  # 6 (benim) + 6 (rakip) + 1 (hamle sırası)
# AlphaZero'nun yaptığı gibi 8x8'lik tahtadan 8x8'lik tahtaya olası tüm hamleler.
# (from_square * 64) + to_square
# Bu, piyon terfilerini tam olarak kapsamaz, ancak utils.py'de bunu 
# "her zaman vezire terfi et" (auto-queen) olarak zorlayacağız.
# Bu, uygulanabilir ve güçlü bir basitleştirmedir.
POLICY_OUTPUT_SIZE = 4096  # 64 * 64

# --- MCTS (Düşünme Süreci) Ayarları ---
# Botun her hamle için ne kadar "düşüneceği" (simülasyon yapacağı).
# Yüksek değer = Daha güçlü oyun, ancak daha yavaş hamle süresi.
MCTS_SIMULATIONS = 25  # İyi bir başlangıç (eğitim için)
                      # Gerçek oyunda (play.py) 400-800 kullanılabilir.

# MCTS'nin "keşif" (exploration) faktörü.
# Yüksek değer = Daha fazla yeni hamle dener.
# Düşük değer = Bildiği en iyi hamleye odaklanır.
CPUCT = 1.5

# --- Kendi Kendine Oynama (Self-Play) Ayarları ---
# Eğitim döngüsü başına oynanacak oyun sayısı.
# Daha fazla oyun = Daha iyi veri, ancak daha yavaş iterasyon.
NUM_SELF_PLAY_GAMES = 100

# Oyunun ilk N hamlesinde, MCTS'nin en çok ziyaret edileni değil,
# ziyaret sayısıyla orantılı bir hamleyi seçmesini sağlar.
# Bu, oyun açılışlarında çeşitliliği artırır (exploration).
EXPLORE_TEMPERATURE_MOVES = 30
# 30 hamleden sonra sıcaklık 0'a yakın olur (en iyi hamleyi seçer - exploitation).

# --- Eğitim (Training) Ayarları ---
# Ana döngünün kaç kez "Eğit -> Oyna -> Veri Topla" yapacağı.
NUM_ITERATIONS = 20

# Toplanan verilerin (deneyimlerin) ne kadarının saklanacağı.
REPLAY_BUFFER_SIZE = 50000  # En son 50 bin hamle verisi

# Eğitim için hafızadan çekilecek veri yığını boyutu.
BATCH_SIZE = 256

# Her yeni veri setiyle sinir ağının kaç kez eğitileceği (epoch).
TRAIN_EPOCHS = 1

# Öğrenme oranı
LEARNING_RATE = 0.001

# --- Model Mimarisi Ayarları ---
# AlphaZero gibi ResNet kullanacağız. Bu, ağdaki "Residual Blok" sayısı.
# Yüksek değer = Daha güçlü model, ancak daha fazla VRAM ve eğitim süresi.
RESIDUAL_BLOCKS = 7 # 19 veya 40 (AlphaZero) çok büyüktür, 7 iyi bir başlangıç.
CONV_FILTERS = 128 # Bloklardaki filtre sayısı

# --- Kayıt Ayarları ---
MODEL_SAVE_PATH = "models/"
BEST_MODEL_NAME = "best_model.pth"

# --- Değerlendirme (Evaluation) Ayarları ---
# Stockfish motorunun bilgisayarınızdaki tam yolu (exe dosyası).
# Windows örneği: "C:/Users/Kullanici/Desktop/stockfish/stockfish_15_x64.exe"
# Linux örneği: "/usr/games/stockfish"
STOCKFISH_PATH = r"c:\Users\samet\Desktop\YAZILIM\Python\SATRANC_RL_PROJESI\stockfish-windows-x86-64-avx2.exe" # Proje kök dizinindeki exe

# Değerlendirme maçlarında kullanılacak Stockfish seviyesi (1-20)
STOCKFISH_SKILL_LEVEL = 10

# --- Loglama Ayarları ---
# Eğitim metriklerini (loss, win rate, duration) kaydetmek için CSV dosyası
TRAIN_LOG_PATH = "training_log.csv"

