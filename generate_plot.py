
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plot():
    # Use the log with more data
    log_path = r"c:\Users\samet\Desktop\YAZILIM\Python\SATRANC_RL_PROJESI\training_log (1).csv"
    
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return

    try:
        df = pd.read_csv(log_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df.index + 1, df['avg_loss'], label='Total Loss', linewidth=2)
        plt.plot(df.index + 1, df['policy_loss'], label='Policy Loss', linestyle='--')
        plt.plot(df.index + 1, df['value_loss'], label='Value Loss', linestyle='--')
        
        plt.title('Eğitim Süreci: Kayıp (Loss) Grafiği')
        plt.xlabel('Adım (Epoch)')
        plt.ylabel('Kayıp (Loss)')
        plt.legend()
        plt.grid(True)
        
        output_path = r"c:\Users\samet\Desktop\YAZILIM\Python\SATRANC_RL_PROJESI\learning_curve.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        
    except Exception as e:
        print(f"Error generating plot: {e}")

if __name__ == "__main__":
    generate_plot()
