import os
import cv2
import numpy as np
import torch
import logging
import requests
import cloudinary
import cloudinary.uploader
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from typing import Optional, Dict, List
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

# Cloudinary Configuration with better error handling
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET")
    )
except Exception as e:
    logger.error(f"Failed to configure Cloudinary: {e}")
    raise

class AttendanceSystem:
    def __init__(self, 
                 yolo_model_path: str = 'yolov8_face.pt',
                 encodings_path: str = 'known_encodings.npy',
                 names_path: str = 'known_names.npy',
                 confidence_threshold: float = 0.79,
                 recognition_threshold: float = 0.6):
        """
        Initialize Attendance System components with type hints and better error handling.
        """
        try:
            # Face Detection Model
            self.face_detector = YOLO(yolo_model_path)
            
            # Face Recognition Setup
            try:
                self.known_encodings = np.load(encodings_path)
                self.known_names = np.load(names_path)
                logger.info(f"Loaded {len(self.known_names)} known faces")
            except Exception as e:
                logger.warning(f"Error loading encodings: {e}. Starting with empty arrays.")
                self.known_encodings = np.array([])
                self.known_names = np.array([])
            
            # Face Embedding Model
            self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.cuda()
                logger.info("Using GPU for face embedding model")
            
            # Thresholds
            self.confidence_threshold = confidence_threshold
            self.recognition_threshold = recognition_threshold
            
        except Exception as e:
            logger.error(f"Failed to initialize AttendanceSystem: {e}")
            raise

    async def upload_to_cloudinary(self, file: UploadFile) -> str:
        """
        Upload image to Cloudinary with improved error handling.
        """
        try:
            file_contents = await file.read()
            upload_result = cloudinary.uploader.upload(
                file_contents,
                folder="attendance_system",
                resource_type="auto"
            )
            logger.info(f"Successfully uploaded image to Cloudinary")
            return upload_result['secure_url']
        except Exception as e:
            logger.error(f"Cloudinary upload error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload image to Cloudinary: {str(e)}"
            )

    def download_image_from_url(self, image_url: str) -> np.ndarray:
        """
        Download image from URL with improved error handling and timeout.
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
                
            return image
        except requests.RequestException as e:
            logger.error(f"Failed to download image: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download image: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error processing downloaded image: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {str(e)}"
            )

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using YOLO with improved error handling.
        """
        try:
            results = self.face_detector(image)
            detected_faces = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_crop = image[y1:y2, x1:x2]
                        face_resized = cv2.resize(face_crop, (160, 160))
                        
                        detected_faces.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'face_image': face_resized
                        })
            
            logger.info(f"Detected {len(detected_faces)} faces in image")
            return detected_faces
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Face detection failed: {str(e)}"
            )

    def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate face embedding with GPU support and error handling.
        """
        try:
            face_tensor = torch.from_numpy(face_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            if torch.cuda.is_available():
                face_tensor = face_tensor.cuda()
            
            with torch.no_grad():
                embedding = self.embedding_model(face_tensor).cpu().squeeze().numpy()
            
            # Normalize embedding
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate face embedding: {str(e)}"
            )

    def recognize_face(self, embedding: np.ndarray) -> Dict:
        """
        Recognize a face based on its embedding with improved response structure.
        """
        try:
            if len(self.known_encodings) == 0:
                return {
                    'recognized': False,
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'message': 'No known faces in database'
                }
            
            distances = [cosine(embedding, known_enc) for known_enc in self.known_encodings]
            min_distance = min(distances)
            best_match_index = distances.index(min_distance)
            
            confidence = 1 - min_distance
            is_recognized = min_distance <= self.recognition_threshold
            
            return {
                'recognized': is_recognized,
                'name': self.known_names[best_match_index] if is_recognized else 'Unknown',
                'confidence': confidence,
                'message': 'Face recognized successfully' if is_recognized else 'No matching face found'
            }
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Face recognition failed: {str(e)}"
            )

# Create FastAPI Application
app = FastAPI(
    title="Attendance Recognition API",
    description="API for face detection and recognition using YOLO and FaceNet",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global attendance system instance
attendance_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the attendance system on startup"""
    global attendance_system
    try:
        attendance_system = AttendanceSystem()
        logger.info("AttendanceSystem initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AttendanceSystem: {e}")
        raise

@app.post("/upload_and_recognize/")
async def upload_and_recognize(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint to upload an image and perform face recognition.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        # Upload to Cloudinary
        cloudinary_url = await attendance_system.upload_to_cloudinary(file)
        
        # Download and process image
        image = attendance_system.download_image_from_url(cloudinary_url)
        detected_faces = attendance_system.detect_faces(image)
        
        # Process each detected face
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
            'status': 'success',
            'faces_detected': len(results),
            'cloudinary_url': cloudinary_url,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API status
    """
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow()}

def main():
    """
    Main function to run the application
    """
    import hypercorn.asyncio
    from hypercorn.config import Config
    
    config = Config()
    port = int(os.getenv("PORT", "8000"))
    config.bind = [f"0.0.0.0:{port}"]
    
    import asyncio
    asyncio.run(hypercorn.asyncio.serve(app, config))

if __name__ == "__main__":
    main()