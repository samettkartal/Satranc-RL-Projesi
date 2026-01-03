# DOSYA: rl_chess/utils.py

import chess
import numpy as np
from typing import Tuple

# Ayar dosyamızdan sabitleri içe aktaralım
from rl_chess import config

# --- Parça -> Kanal Eşlemesi ---
# Bu, board_to_tensor için gereklidir
# 6 beyaz parça (0-5), 6 siyah parça (6-11)
PIECE_TO_CHANNEL_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # Beyaz parçalar
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # Siyah parçalar
}
# Kanal 12: Hamle sırası (hep 1.0)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Bir chess.Board nesnesini (config.INPUT_CHANNELS, 8, 8) boyutlu bir 
    numpy dizisine dönüştürür.
    
    Ağ, tahtayı HER ZAMAN mevcut oyuncunun bakış açısından görür.
    Sıra siyahtaysa, tahta (ve parça kanalları) ters çevrilir.
    """
    
    # 1. Bakış açısını (perspektifi) ayarla
    if board.turn == chess.WHITE:
        perspective = chess.WHITE
    else:
        perspective = chess.BLACK

    # Boş tensörü başlat
    # (13, 8, 8)
    tensor = np.zeros(
        (config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE), 
        dtype=np.float32
    )

    # 2. Parçaları kanallara yerleştir
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        if piece:
            # Satır ve sütunu al
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            # Perspektife göre satırı dönüştür (siyahsa ters çevir)
            if perspective == chess.BLACK:
                rank = 7 - rank # Satırı ters çevir
                file = 7 - file # Sütunu ters çevir (opsiyonel ama tutarlı)

            # Parçanın rengine göre doğru kanal grubunu seç
            if piece.color == perspective:
                # "Benim" parçalarım (0-5 arası kanallar)
                channel = PIECE_TO_CHANNEL_MAP[piece.symbol().upper()]
            else:
                # "Rakip" parçaları (6-11 arası kanallar)
                channel = PIECE_TO_CHANNEL_MAP[piece.symbol().lower()]
                
            tensor[channel, rank, file] = 1

    # 3. Son kanalı (12. indeks) hamle sırası bilgisiyle doldur
    # Bu her zaman 1.0'dır, çünkü "şu anki oyuncunun" sırası olduğunu belirtir.
    tensor[12, :, :] = 1.0 
    
    # Rok haklarını, en-passant'ı vb. ek kanallar olarak eklemek 
    # modeli güçlendirir, ancak 13 kanallı bu yapı güçlü bir başlangıçtır.
    
    return tensor

def move_to_index(move: chess.Move) -> int:
    """
    Bir chess.Move nesnesini 0-4095 arası bir tam sayıya dönüştürür.
    
    ÖNEMLİ: Bu fonksiyon piyon terfilerini (promotion) KAPSAMAZ.
    Piyon terfileri (örn. e7e8q), temel 'from_square' ve 'to_square' 
    bilgisiyle aynı indekse eşlenir. MCTS, yasal olup olmadığını 
    kontrol ederken terfi hamlesini ayrıca değerlendirir.
    
    Format: from_square * 64 + to_square
    """
    return move.from_square * 64 + move.to_square

def index_to_move_tuple(index: int) -> Tuple[int, int]:
    """
    Bir indeksi (from_square, to_square) demetine dönüştürür.
    Bu, MCTS'de hızlı kontrol için kullanılır.
    """
    to_square = index % 64
    from_square = index // 64
    return (from_square, to_square)

def get_legal_moves_mask(board: chess.Board) -> np.ndarray:
    """
    Mevcut tahta durumu için yasal hamleleri gösteren 
    (config.POLICY_OUTPUT_SIZE,) = (4096,) boyutlu bir maske (boolean array) oluşturur.
    
    Maskede 'True' olan indeksler, yasal bir hamleye karşılık gelir.
    Bu, MCTS'nin ve ağın yasadışı hamleleri seçmesini engellemek için KRİTİKTİR.
    """
    mask = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=bool)
    
    # Siyahın perspektifindeysek, hamleleri de ters çevirmeliyiz.
    perspective = board.turn
    
    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        
        if perspective == chess.BLACK:
            # Hamleleri de siyahın bakış açısına göre "ters çevir"
            from_sq = 63 - from_sq # (7-rank)*8 + (7-file)
            to_sq = 63 - to_sq
            
        # Piyon terfisi durumunu ele al (auto-queen)
        # Bu, e7e8, e7e8q, e7e8r... hepsinin aynı indekse gitmesini sağlar.
        # Bizim config'imiz (4096) sadece from/to'ya dayalıdır.
        # Eğer hamle bir terfi ise, onu 'auto-queen' olarak kabul ederiz.
        # chess.Move'un kendisi terfiyi zaten içerir.
        
        index = from_sq * 64 + to_sq
        mask[index] = True
            
    return mask

# --- Test Bloğu ---
if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırıldığında testleri yap
    
    # 1. board_to_tensor testi
    b = chess.Board()
    tensor = board_to_tensor(b)
    print(f"Başlangıç tahtası tensör boyutu: {tensor.shape}")
    assert tensor.shape == (config.INPUT_CHANNELS, 8, 8)
    # Beyaz piyonlar (Kanal 0), Satır 1 (indeks 1)
    assert np.sum(tensor[0, 1, :]) == 8 
    # Siyah piyonlar (Kanal 6), Satır 6 (indeks 6)
    assert np.sum(tensor[6, 6, :]) == 8
    
    # 2. Siyah perspektifi testi
    b.push_san("e4") # Sıra siyahta
    tensor_black = board_to_tensor(b)
    # Artık 'benim piyonlarım' (Kanal 0) siyahın piyonlarıdır.
    # Ve tahta ters çevrildiği için 'rank 6' (indeks 6) yerine 
    # 'rank 1' (indeks 1) olarak görünmelidirler.
    assert np.sum(tensor_black[0, 1, :]) == 8 
    # 'Rakip piyonları' (Kanal 6) beyazın piyonlarıdır.
    # e4'teki piyon (rank 3) artık (7-3) = rank 4'te görünmelidir.
    assert tensor_black[6, 4, chess.E4 % 8] == 1 
    print("Perspektif testi başarılı.")

    # 3. Maske testi
    b = chess.Board()
    mask = get_legal_moves_mask(b)
    print(f"Başlangıç maske boyutu: {mask.shape}")
    assert mask.shape == (config.POLICY_OUTPUT_SIZE,)
    # Başlangıçta 20 yasal hamle var
    assert np.sum(mask) == 20 
    print(f"Başlangıç hamle sayısı (maske): {np.sum(mask)}")

    # e2e4 hamlesinin maskede olması lazım
    e2e4_move = chess.Move.from_uci("e2e4")
    e2e4_index = e2e4_move.from_square * 64 + e2e4_move.to_square
    assert mask[e2e4_index] == True
    
    # e2e5 hamlesinin maskede olmaması lazım
    e2e5_move = chess.Move.from_uci("e2e5")
    e2e5_index = e2e5_move.from_square * 64 + e2e5_move.to_square
    assert mask[e2e5_index] == False
    
    print("Maske testleri başarılı.")
    print("utils.py başarıyla oluşturuldu ve test edildi.")