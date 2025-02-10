import os
import requests

def download_model(model_url, save_path):
    if not os.path.exists(save_path):
        print("Downloading the model...")
        response = requests.get(model_url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("Model already exists.")

# Example usage
if __name__ == "__main__":
    MODEL_URL = "/Users/aashiqmahato/Downloads/project/Facenet recognization/yolov8_face.pt"
    SAVE_PATH = "yolov8_face.pt"
    download_model(MODEL_URL, SAVE_PATH)