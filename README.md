# Satranç Pekiştirmeli Öğrenme (Reinforcement Learning) Projesi

Bu proje, **AlphaZero** algoritmasından esinlenerek geliştirilmiş, kendi kendine oynayarak (self-play) öğrenen bir satranç yapay zekasıdır.

Referans Rapor: Raporun tamamı için [RAPOR.md](RAPOR.md) veya `pekiştirmeliöğrenme (2).pdf` dosyasına bakabilirsiniz.

## Özellikler
- **Model Tabanlı RL:** Monte Carlo Tree Search (MCTS) ve Derin Sinir Ağları (ResNet) kombinasyonu.
- **Kendi Kendine Öğrenme:** İnsan verisine ihtiyaç duymadan, sıfırdan satranç oynamayı öğrenir.
- **Pygame Arayüzü:** Eğitilmiş modele karşı oynamak için görsel arayüz.
- **Paralel Eğitim:** Çok çekirdekli işlemcilerde hızlandırılmış veri toplama süreci.

## Kurulum

Projeyi çalıştırmak için Python 3.8+ ve aşağıdaki kütüphanelere ihtiyacınız vardır:

```bash
pip install -r requirements.txt
```

*Gereken kütüphaneler: `torch`, `numpy`, `python-chess`, `pygame`*

## Kullanım

### 1. Eğitilmiş Modele Karşı Oyna (GUI)

Proje `models/best_model.pth` dosyasında eğitilmiş bir model içerir. Bu modele karşı oynamak için:

```bash
python gui_pygame.py
```

- **Kontroller:** Fare ile taşları sürükleyip bırakabilir veya tıklayarak hareket ettirebilirsiniz.
- **Oyun:** Siz Beyaz taşlarla oynarsınız, yapay zeka Siyah taşlarla oynar.

### 2. Modeli Eğitmek (Training)

Modeli sıfırdan eğitmek veya mevcut eğitimi devam ettirmek için:

```bash
python main.py
```

Bu komut:
1. Self-play ile oyun verisi toplar.
2. Sinir ağını eğitir.
3. Modeli değerlendirir (vs Random veya vs Stockfish).
4. En iyi modeli `models/` klasörüne kaydeder.

## Proje Yapısı

- `rl_chess/`: Algoritma çekirdek dosyaları (MCTS, Network, Self-Play vb.).
- `models/`: Kaydedilmiş modeller. (Github'da sadece en iyi model saklanır).
- `assets/`: Satranç taşları ve arayüz görselleri.
- `main.py`: Eğitim döngüsü giriş noktası.
- `gui_pygame.py`: Oyun arayüzü.

## Rapor Özeti

### Algoritma
Proje, her pozisyon için "Kazanma Olasılığı" (Value) ve "En İyi Hamle Olasılıkları"nı (Policy) tahmin eden bir **ResNet** mimarisi kullanır. MCTS (Monte Carlo Tree Search), bu tahminleri kullanarak geleceğe yönelik simülasyonlar yapar ve en iyi hamleyi seçer.

### Eğitim Sonuçları
4 saatlik (A100 GPU) eğitim sonucunda model:
- Kuralları tam olarak öğrenmiştir.
- Rastgele oynayan botları %100 oranında yenmektedir.
- Basit taktikleri ve mat kalıplarını uygulayabilmektedir.
- Loss (Hata) değeri 7.3'ten 1.15'e düşmüştür.

Eğitim grafikleri:
![Eğitim Yakınsama](training_convergence_graph.png)

---
*Geliştirici Notu: Bu proje eğitim amaçlıdır ve AlphaZero'nun basitleştirilmiş bir implementasyonudur.*
