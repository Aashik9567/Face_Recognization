import os
import cv2
import numpy as np
import torch
import logging
import requests
import cloudinary
import cloudinary.uploader
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optionally, download the model if not present
MODEL_PATH = "yolov8_face.pt"
MODEL_URL = os.environ.get("YOLO_MODEL_URL")
if not os.path.exists(MODEL_PATH) and MODEL_URL:
    from model_loader import download_model
    print("Model file not found. Downloading it...")
    download_model(MODEL_URL, MODEL_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)

# Cloudinary Configuration using environment variables
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "aashiqmahato"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", "822293972657394"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "yGUpxVroCkjj40nThHOv56u2CZM")
)

class AttendanceSystem:
    def __init__(self, 
                 yolo_model_path=MODEL_PATH, 
                 encodings_path='known_encodings.npy', 
                 names_path='known_names.npy', 
                 confidence_threshold=0.79,
                 recognition_threshold=0.6):
        """
        Initialize Attendance System components.
        Note: The yolov8_face.pt file is huge, so consider using Git LFS or loading it from an external source.
        """
        # Face Detection Model
        self.face_detector = YOLO(yolo_model_path)
        
        # Face Recognition Setup
        try:
            self.known_encodings = np.load(encodings_path)
            self.known_names = np.load(names_path)
            logging.info(f"Loaded {len(self.known_names)} known faces")
        except Exception as e:
            logging.error(f"Error loading encodings: {e}")
            self.known_encodings = np.array([])
            self.known_names = np.array([])
        
        # Face Embedding Model
        self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.recognition_threshold = recognition_threshold

    def upload_to_cloudinary(self, file: UploadFile):
        """
        Upload image to Cloudinary.
        """
        try:
            file_contents = file.file.read()
            upload_result = cloudinary.uploader.upload(
                file_contents, 
                folder="attendance_system"
            )
            return upload_result['secure_url']
        except Exception as e:
            logging.error(f"Cloudinary upload error: {e}")
            raise HTTPException(status_code=500, detail=f"Upload to Cloudinary failed: {str(e)}")

    def download_image_from_url(self, image_url):
        """
        Download image from URL.
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            return image
        except Exception as e:
            logging.error(f"Image download error: {e}")
            raise

    def detect_faces(self, image):
        """
        Detect faces in an image using YOLO.
        """
        results = self.face_detector(image)
        detected_faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    face_crop = image[y1:y2, x1:x2]
                    face_resized = cv2.resize(face_crop, (160, 160))
                    detected_faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'face_image': face_resized
                    })
        return detected_faces

    def generate_embedding(self, face_image):
        """
        Generate face embedding.
        """
        face_tensor = torch.from_numpy(face_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            embedding = self.embedding_model(face_tensor).squeeze().numpy()
        # Normalize the embedding
        return embedding / np.linalg.norm(embedding)

    def recognize_face(self, embedding):
        """
        Recognize a face based on its embedding.
        """
        if len(self.known_encodings) == 0:
            return {
                'recognized': False,
                'name': 'Unknown',
                'confidence': 0.0
            }
        distances = [cosine(embedding, known_enc) for known_enc in self.known_encodings]
        min_distance = min(distances)
        best_match_index = distances.index(min_distance)
        if min_distance <= self.recognition_threshold:
            return {
                'recognized': True,
                'name': self.known_names[best_match_index],
                'confidence': 1 - min_distance
            }
        else:
            return {
                'recognized': False,
                'name': 'Unknown',
                'confidence': 1 - min_distance
            }

# Create FastAPI Application
app = FastAPI(title="Attendance Recognition API")

# Add CORS middleware to allow all origins (you may restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global attendance system instance
attendance_system = AttendanceSystem()

@app.post("/upload_and_recognize/")
async def upload_and_recognize(file: UploadFile = File(...)):
    """
    Endpoint to upload an image, process it through Cloudinary, and run face recognition.
    """
    if not file:
        logging.error("No file received in the request.")
        raise HTTPException(status_code=400, detail="File is missing.")
    try:
        cloudinary_url = attendance_system.upload_to_cloudinary(file)
        image = attendance_system.download_image_from_url(cloudinary_url)
        detected_faces = attendance_system.detect_faces(image)
        results = []
        for face_data in detected_faces:
            embedding = attendance_system.generate_embedding(face_data['face_image'])
            recognition_result = attendance_system.recognize_face(embedding)
            result = {
                'bbox': face_data['bbox'],
                'detection_confidence': face_data['confidence'],
                'cloudinary_url': cloudinary_url,
                **recognition_result
            }
            results.append(result)
        return JSONResponse(content={
            'faces_detected': len(results),
            'cloudinary_url': cloudinary_url,
            'results': results
        })
    except Exception as e:
        logging.error(f"Face recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Entry point when running locally
if __name__ == '__main__':
    import hypercorn.asyncio
    from hypercorn.config import Config
    config = Config()
    # Bind to the $PORT variable if set, otherwise default to 8000
    port = os.environ.get("PORT", "8000")
    config.bind = [f"0.0.0.0:{port}"]
    import asyncio
    asyncio.run(hypercorn.asyncio.serve(app, config))