import zipfile
import os

def create_bundle():
    zip_name = "project_bundle.zip"
    files_to_include = [
        "main.py",
        "training_log.csv"
    ]
    folders_to_include = [
        "rl_chess",
        "models"
    ]
    
    print(f"'{zip_name}' oluşturuluyor...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Tekil dosyaları ekle
        for file in files_to_include:
            if os.path.exists(file):
                print(f"Ekleniyor: {file}")
                zipf.write(file)
            else:
                print(f"UYARI: {file} bulunamadı.")
                
        # Klasörleri ekle
        for folder in folders_to_include:
            if os.path.exists(folder):
                print(f"Klasör ekleniyor: {folder}")
                for root, dirs, files in os.walk(folder):
                    # __pycache__ klasörlerini atla
                    if "__pycache__" in root:
                        continue
                        
                    for file in files:
                        if file.endswith(".pyc"): continue
                        
                        file_path = os.path.join(root, file)
                        # Zip içindeki yol (root'a göre göreceli)
                        arcname = os.path.relpath(file_path, ".")
                        print(f"  -> {arcname}")
                        zipf.write(file_path, arcname)
            else:
                print(f"UYARI: {folder} klasörü bulunamadı.")
                
    print(f"\nTamamlandı! '{zip_name}' dosyası hazır.")

if __name__ == "__main__":
    create_bundle()
