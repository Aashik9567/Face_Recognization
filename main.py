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
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)

# Cloudinary Configuration
cloudinary.config(
    cloud_name="aashiqmahato",  
    api_key="822293972657394",        
    api_secret="yGUpxVroCkjj40nThHOv56u2CZM"  
)

class AttendanceSystem:
    def __init__(self, 
                 yolo_model_path='/Users/aashiqmahato/Downloads/project/Face detection api/yolov8_face.pt', 
                 encodings_path='known_encodings.npy', 
                 names_path='known_names.npy', 
                 confidence_threshold=0.79,
                 recognition_threshold=0.6):
        """
        Initialize Attendance System components
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

    def upload_to_cloudinary(self, file):
        """
        Upload image to Cloudinary
        
        Args:
            file (UploadFile): Image file to upload
        
        Returns:
            str: Cloudinary URL of the uploaded image
        """
        try:
            # Read file contents
            file_contents = file.file.read()
            
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                file_contents, 
                folder="attendance_system"
            )
            
            # Return secure URL
            return upload_result['secure_url']
        except Exception as e:
            logging.error(f"Cloudinary upload error: {e}")
            raise HTTPException(status_code=500, detail=f"Upload to Cloudinary failed: {str(e)}")

    def download_image_from_url(self, image_url):
        """
        Download image from URL
        
        Args:
            image_url (str): URL of the image
        
        Returns:
            np.ndarray: Image as numpy array
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Convert to numpy array
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
        Detect faces in an image using YOLO
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            list: Detected faces with coordinates and confidence
        """
        results = self.face_detector(image)
        
        detected_faces = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                confidence = box.conf[0]
                
                # Apply confidence threshold
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Crop face
                    face_crop = image[y1:y2, x1:x2]
                    
                    # Resize for embedding
                    face_resized = cv2.resize(face_crop, (160, 160))
                    
                    detected_faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'face_image': face_resized
                    })
        
        return detected_faces

    def generate_embedding(self, face_image):
        """
        Generate face embedding
        
        Args:
            face_image (np.ndarray): Preprocessed face image
        
        Returns:
            np.ndarray: Face embedding
        """
        # Preprocess for PyTorch
        face_tensor = torch.from_numpy(face_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.embedding_model(face_tensor).squeeze().numpy()
        
        # Normalize embedding
        return embedding / np.linalg.norm(embedding)

    def recognize_face(self, embedding):
        """
        Recognize a face based on its embedding
        
        Args:
            embedding (np.ndarray): Face embedding
        
        Returns:
            dict: Recognition result
        """
        if len(self.known_encodings) == 0:
            return {
                'recognized': False,
                'name': 'Unknown',
                'confidence': 0.0
            }
        
        # Calculate cosine distances
        distances = [cosine(embedding, known_enc) for known_enc in self.known_encodings]
        
        # Find best match
        min_distance = min(distances)
        best_match_index = distances.index(min_distance)
        
        # Check if within recognition threshold
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

# FastAPI Application
app = FastAPI(title="Attendance Recognition API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global attendance system instance
attendance_system = AttendanceSystem()

@app.post("/upload_and_recognize/")
async def upload_and_recognize(file: UploadFile = File(...)):
    """
    Endpoint to upload image to Cloudinary and recognize faces
    """
    if not file:
        logging.error("No file received in the request.")
        raise HTTPException(status_code=400, detail="File is missing.")
    
    try:
        # Upload image to Cloudinary
        cloudinary_url = attendance_system.upload_to_cloudinary(file)
        
        # Download image from Cloudinary URL
        image = attendance_system.download_image_from_url(cloudinary_url)
        
        # Detect faces
        detected_faces = attendance_system.detect_faces(image)
        
        # Recognize faces
        results = []
        for face_data in detected_faces:
            # Generate embedding
            embedding = attendance_system.generate_embedding(face_data['face_image'])
            
            # Recognize face
            recognition_result = attendance_system.recognize_face(embedding)
            
            # Combine detection and recognition results
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

# Run the application with Hypercorn
def main():
    import hypercorn.asyncio
    from hypercorn.config import Config
    
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    
    import asyncio
    asyncio.run(hypercorn.asyncio.serve(app, config))

if __name__ == '__main__':
    main()