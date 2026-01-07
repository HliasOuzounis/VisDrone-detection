import gdown
import os

def download_mot_test():
    files = {
        "MOT_Test_Dev": "14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu", # Example ID
    }
    
    output_dir = "data_test/VisDrone2019-MOT-test-dev"
    os.makedirs(output_dir, exist_ok=True)

    for name, file_id in files.items():
        url = f'https://drive.google.com/uc?id={file_id}'
        output_path = os.path.join(output_dir, f"{name}.zip")
        
        print(f"Starting download: {name}")
        gdown.download(url, output_path, quiet=False)
        
        import zipfile
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted to {output_dir}")

if __name__ == "__main__":
    download_mot_test()