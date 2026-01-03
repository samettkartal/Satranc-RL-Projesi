# DOSYA: rl_chess/mcts.py

import math
import numpy as np
import torch
import chess
from typing import Dict, Optional, Tuple
import torch.nn.functional as F

from rl_chess import config, utils, network

class Node:
    """
    Arama ağacındaki tek bir pozisyonu (düğümü) temsil eder.
    
    Her düğüm, o pozisyona ulaşıldığında MCTS tarafından toplanan 
    istatistikleri (ziyaret sayısı, toplam değer vb.) tutar.
    """
    def __init__(self, parent: Optional['Node'], board: chess.Board, prior_p: float):
        self.parent = parent
        self.board = board  # Bu düğümdeki satranç tahtası durumu
        
        # --- MCTS İstatistikleri ---
        # N(s,a): Bu düğümün (veya bu düğüme yol açan eylemin) ziyaret sayısı
        self.N: int = 0
        # W(s,a): Bu düğümden elde edilen toplam değer (simülasyon sonuçları)
        self.W: float = 0.0
        # Q(s,a): Ortalama değer (W / N)
        # self.Q() metodu ile hesaplanır
        
        # P(s,a): Sinir ağından gelen öncelikli olasılık (Prior Probability)
        # Bu eylemin (hamlenin) "sezgisel" olarak ne kadar iyi olduğu.
        self.P = prior_p
        
        # --- Çocuk Düğümler ---
        # Bu düğümden kaynaklanan yasal hamleler ve onların
        # sonuç düğümleri (children).
        # Dict[chess.Move, Node]
        self.children: Dict[chess.Move, 'Node'] = {}
        self.is_expanded: bool = False

    def Q(self) -> float:
        """
        Bu düğümün ortalama eylem değerini (Q-value) döndürür.
        Q(s, a) = W(s, a) / N(s, a)
        Eğer hiç ziyaret edilmemişse, 0 döndür (tarafsız).
        """
        if self.N == 0:
            return 0.0
        # Değer, her zaman *mevcut oyuncunun* bakış açısındandır.
        return self.W / self.N

    def select(self) -> 'Node':
        """
        PUCT formülünü kullanarak bu düğümün çocuklarından en iyi 
        (en umut verici) olanı seçer.
        """
        best_child = None
        best_puct = -float('inf')
        
        # Ziyaret edilen tüm alt düğümlerin toplam N'sinin karekökü
        # Keşif (exploration) için kullanılır.
        sqrt_total_N = math.sqrt(sum(child.N for child in self.children.values()))

        for move, child in self.children.items():
            # AlphaZero PUCT (Polynomial Upper Confidence Trees) formülü:
            # PUCT = Q(s,a) + C_puct * P(s,a) * (sqrt(sum(N(s,b))) / (1 + N(s,a)))
            # Q(s,a): Mevcut bilinen değer (exploitation - sömürü)
            # P(s,a): Ağa göre "sezgisel" değer (exploration - keşif)
            
            # self.Q() (çocuğun değeri) zaten ebeveynin (bizim)
            # bakış açısına göre -1 ile çarpılmıştır (backpropagate'de).
            # Bu yüzden burada Q() değerini doğrudan kullanırız.
            puct_score = child.Q() + config.CPUCT * child.P * \
                         (sqrt_total_N / (1 + child.N))

            if puct_score > best_puct:
                best_puct = puct_score
                best_child = child
                
        return best_child

    def expand(self, policy_probs: np.ndarray, legal_mask: np.ndarray):
        """
        Bir yaprak (leaf) düğümü genişletir.
        Sinir ağından gelen politika (policy_probs) ve yasal hamle 
        maskesini (legal_mask) kullanarak tüm yasal hamleler için
        çocuk düğümler (child nodes) oluşturur.
        """
        self.is_expanded = True
        
        # Yasal hamleleri al
        for move in self.board.legal_moves:
            if move not in self.children:
                # Ağın bu hamleye atadığı "öncelikli" olasılığı al
                # utils.py'deki perspektif dönüşümünü burada da uygulamalıyız.
                
                from_sq = move.from_square
                to_sq = move.to_square
                
                # Eğer sıra siyahtaysa, ağı beslerken hamleleri
                # ters çevirmiştik (utils.py'de). Burada da aynı 
                # dönüşümü indeksi bulmak için yapmalıyız.
                if self.board.turn == chess.BLACK:
                    from_sq = 63 - from_sq
                    to_sq = 63 - to_sq
                    
                move_index = from_sq * 64 + to_sq
                
                # Maske kontrolü (özellikle piyon terfisi için önemli)
                if legal_mask[move_index]:
                    # Yeni tahta durumunu oluştur
                    next_board = self.board.copy()
                    next_board.push(move)
                    
                    # Çocuk düğümü oluştur ve sinir ağından gelen
                    # hamle olasılığını (prior_p) ata
                    self.children[move] = Node(
                        parent=self, 
                        board=next_board, 
                        prior_p=policy_probs[move_index]
                    )

    def backpropagate(self, value: float):
        """
        Simülasyon sonucunu (value) ağaçta yukarı doğru (ebeveynlere) yayar.
        """
        node = self
        # Değeri her zaman *ebeveynin* bakış açısına göre çeviririz.
        # Eğer ben (çocuk) kazandıysam (+1), ebeveyn (rakip) 
        # o hamleyi yaparak kaybetmiş (-1) olur.
        current_value = -value 
        
        while node is not None:
            node.N += 1
            node.W += current_value
            # Bir üst ebeveyne geç
            node = node.parent
            # Değeri bir sonraki (üstteki) ebeveyn için tekrar ters çevir
            current_value = -current_value

class MCTS:
    """
    Monte Carlo Ağaç Aramasını yöneten ana sınıf.
    """
    def __init__(self, model: network.PolicyValueNet):
        self.model = model
        self.model.eval() # MCTS her zaman 'inference' modundadır

    def run_simulation(self, root_node: Node):
        """
        Kökten (root) başlayarak tek bir MCTS simülasyonu (select, 
        expand/evaluate, backpropagate) gerçekleştirir.
        """
        
        node = root_node
        
        # --- 1. SEÇİM (SELECT) ---
        # Oyun sonu olmayan ve genişletilmiş (expanded) düğümler 
        # arasında dolaş.
        while node.is_expanded and not node.board.is_game_over():
            node = node.select()

        # --- 2. GENİŞLETME VE DEĞERLENDİRME (EXPAND & EVALUATE) ---
        # Artık bir yaprak (leaf) düğümdeyiz (veya oyun sonu).
        
        # Oyunun bu yaprakta bitip bitmediğini kontrol et
        if node.board.is_game_over():
            result = node.board.result(claim_draw=True)
            if result == '1-0': # Beyaz kazandı
                value = 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == '0-1': # Siyah kazandı
                value = 1.0 if node.board.turn == chess.WHITE else -1.0
            else: # Beraberlik (1/2-1/2, *, vb.)
                value = 0.0
            
            # Değer, *mevcut* oyuncunun (node.board.turn) 
            # bakış açısındandır.
            # (Eğer sıra siyahtaysa ve beyaz kazandıysa (1-0), 
            # siyahın değeri -1.0'dir)
        
        else:
            # Oyun bitmediyse, "beyne" (sinir ağına) danış
            
            # Tahtayı sinir ağının anlayacağı tensöre dönüştür
            board_tensor = torch.from_numpy(
                utils.board_to_tensor(node.board)
            ).unsqueeze(0).to(config.DEVICE) # (1, C, H, W)
            
            with torch.no_grad():
                policy_logits, value_tensor = self.model(board_tensor)
            
            # Değeri al (scalar)
            value = value_tensor.item() # [-1, 1] arası
            
            # Politikayı (hamle olasılıklarını) al
            # Logit'leri olasılıklara çevir (Softmax)
            policy_logits = policy_logits.squeeze(0) # (4096,)
            
            # Yasal hamle maskesini al
            legal_mask = utils.get_legal_moves_mask(node.board)
            
            # Yasal olmayan hamlelerin olasılığını -inf yap (softmax'tan önce)
            policy_logits[~legal_mask] = -float('Inf')
            
            # Yasal hamleler üzerinde Softmax uygula
            policy_probs = F.softmax(policy_logits, dim=0).cpu().numpy()
            
            # Düğümü bu politikayı kullanarak genişlet
            node.expand(policy_probs, legal_mask)
            
            # Değer, *mevcut* oyuncunun (node.board.turn) 
            # bakış açısındandır.
            
        # --- 3. GERİ YAYILIM (BACKPROPAGATE) ---
        # Elde edilen 'value'yu (ister oyun sonu, ister ağ tahmini)
        # ağaçta yukarı doğru yay.
        node.backpropagate(value)

    def get_best_move(self, board: chess.Board, 
                      temperature: float) -> Tuple[chess.Move, np.ndarray, Node]:
        """
        Belirli bir tahta durumu için MCTS simülasyonlarını çalıştırır 
        ve en iyi hamleyi döndürür.
        
        Ayrıca, eğitim için 'politika hedefi' (pi) vektörünü de döndürür.
        """
        
        # Kök düğümü oluştur
        root_node = Node(parent=None, board=board.copy(), prior_p=0.0)

        # Kök düğümü bir kez değerlendirip genişletmemiz gerekiyor
        # (run_simulation bunu yapmaz, select ile başlar)
        if not root_node.board.is_game_over():
            board_tensor = torch.from_numpy(
                utils.board_to_tensor(root_node.board)
            ).unsqueeze(0).to(config.DEVICE)
            
            with torch.no_grad():
                policy_logits, _ = self.model(board_tensor)
                
            policy_logits = policy_logits.squeeze(0)
            legal_mask = utils.get_legal_moves_mask(root_node.board)
            policy_logits[~legal_mask] = -float('Inf')
            policy_probs = F.softmax(policy_logits, dim=0).cpu().numpy()
            
            root_node.expand(policy_probs, legal_mask)
        
        # Yapılandırmada belirtilen sayıda simülasyonu çalıştır
        for _ in range(config.MCTS_SIMULATIONS):
            self.run_simulation(root_node)
            
        # --- Hamle Seçimi ve Eğitim Politikası (pi) ---
        
        # Eğitim için politika hedefi (pi) vektörünü oluştur
        # Bu, MCTS'nin "düşünülmüş" politikasıdır.
        # Hedef: Sinir ağının ham tahmini (P), bu 'pi'ye yaklaşsın.
        pi = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
        
        child_moves = []
        visit_counts = []
        
        if not root_node.children:
            # Bu, mat/pat durumu olabilir, döndürülecek hamle yok
            return None, pi, root_node
            
        for move, child in root_node.children.items():
            child_moves.append(move)
            visit_counts.append(child.N)
            
            # 'pi' vektörünü doldur
            from_sq = move.from_square
            to_sq = move.to_square
            if root_node.board.turn == chess.BLACK:
                from_sq = 63 - from_sq
                to_sq = 63 - to_sq
            move_index = from_sq * 64 + to_sq
            pi[move_index] = child.N
            
        # Ziyaret sayılarını (visit counts) 'pi' politikasına dönüştür
        if np.sum(visit_counts) > 0:
            pi = pi / np.sum(visit_counts) # Normalleştir
        
        # --- Oynanacak Hamleyi Seç ---
        if temperature == 0:
            # Sıcaklık 0 ise (deterministic), en çok ziyaret edileni seç (exploitation)
            best_move_index = np.argmax(visit_counts)
            best_move = child_moves[best_move_index]
        else:
            # Sıcaklık > 0 ise (stochastic), ziyaret sayılarına göre 
            # olasılıksal bir seçim yap (exploration)
            # Sıcaklık, dağılımı "yumuşatır" veya "keskinleştirir"
            probs = np.array(visit_counts)**(1.0 / temperature)
            probs = probs / np.sum(probs) # Normalleştir
            
            best_move = np.random.choice(child_moves, p=probs)
            
        return best_move, pi, root_node