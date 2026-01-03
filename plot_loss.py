import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_presentation_graphs(log_file):
    if not os.path.exists(log_file):
        print(f"Hata: Dosya bulunamadı - {log_file}")
        return

    try:
        # CSV dosyasını oku
        df = pd.read_csv(log_file)
        df['iteration'] = range(1, len(df) + 1)
        
        # 1. Grafik: Eğitim Yakınsama Eğrisi (Total Loss)
        plt.figure(figsize=(10, 6))
        plt.plot(df['iteration'], df['avg_loss'], marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Toplam Kayıp (Total Loss)')
        plt.title('Eğitim Yakınsama Eğrisi (Training Convergence)', fontsize=16, fontweight='bold')
        plt.xlabel('İterasyon (Iteration)', fontsize=12)
        plt.ylabel('Ortalama Kayıp (Average Loss)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('training_convergence_graph.png', dpi=300)
        print("1. Grafik oluşturuldu: training_convergence_graph.png")
        plt.close()

        # 2. Grafik: Politika Gelişimi (Policy Loss)
        plt.figure(figsize=(10, 6))
        plt.plot(df['iteration'], df['policy_loss'], marker='s', linestyle='-', color='#d62728', linewidth=2, label='Politika Kaybı (Policy Loss)')
        plt.title('Politika Ağı Optimizasyonu (Policy Network Optimization)', fontsize=16, fontweight='bold')
        plt.xlabel('İterasyon (Iteration)', fontsize=12)
        plt.ylabel('Politika Kaybı (Policy Loss)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('policy_improvement_graph.png', dpi=300)
        print("2. Grafik oluşturuldu: policy_improvement_graph.png")
        plt.close()
        
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    log_filename = "training_log (1).csv"
    plot_presentation_graphs(log_filename)
