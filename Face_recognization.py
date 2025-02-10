import numpy as np
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
import logging
from scipy.spatial.distance import cosine

class FaceRecognizer:
    def __init__(self, 
                 encodings_path='known_encodings.npy', 
                 names_path='known_names.npy',
                 threshold=0.6):
        """
        Initialize Face Recognizer
        
        Args:
            encodings_path (str): Path to saved face encodings
            names_path (str): Path to saved face names
            threshold (float): Distance threshold for face matching
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        
        # Load known face encodings and names
        try:
            self.known_encodings = np.load(encodings_path)
            self.known_names = np.load(names_path)
            
            logging.info(f"Loaded {len(self.known_names)} known faces")
        except Exception as e:
            logging.error(f"Error loading encodings: {e}")
            raise
        
        # Recognition threshold
        self.threshold = threshold
        
        # Initialize FaceNet model for embedding generation
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def preprocess_image(self, image_path):
        """
        Preprocess image for face detection and embedding
        
        Args:
            image_path (str): Path to input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to FaceNet input size
            img = cv2.resize(img, (160, 160))
            
            # Normalize and convert to tensor
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            
            return img
        
        except Exception as e:
            logging.error(f"Image preprocessing error: {e}")
            return None

    def generate_embedding(self, image_path):
        """
        Generate embedding for a single image
        
        Args:
            image_path (str): Path to input image
        
        Returns:
            np.ndarray: Face embedding
        """
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            if img_tensor is None:
                return None
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(img_tensor).squeeze().numpy()
            
            # Normalize embedding
            return embedding / np.linalg.norm(embedding)
        
        except Exception as e:
            logging.error(f"Embedding generation error: {e}")
            return None

    def recognize_face(self, image_path):
        """
        Recognize face in the given image
        
        Args:
            image_path (str): Path to input image
        
        Returns:
            dict: Recognition results
        """
        try:
            # Generate embedding for input image
            input_embedding = self.generate_embedding(image_path)
            
            if input_embedding is None:
                return {"error": "Could not generate embedding"}
            
            # Calculate distances to known faces
            distances = [cosine(input_embedding, known_enc) for known_enc in self.known_encodings]
            
            # Find the best match
            min_distance = min(distances)
            best_match_index = distances.index(min_distance)
            
            # Determine if the face is recognized
            if min_distance <= self.threshold:
                recognized_name = self.known_names[best_match_index]
                return {
                    "recognized": True,
                    "name": recognized_name,
                    "confidence": 1 - min_distance
                }
            else:
                return {
                    "recognized": False,
                    "name": "Unknown",
                    "confidence": 1 - min_distance
                }
        
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            return {"error": str(e)}

    def batch_recognize(self, image_paths):
        """
        Recognize faces in multiple images
        
        Args:
            image_paths (list): List of image paths
        
        Returns:
            list: Recognition results for each image
        """
        results = []
        for image_path in image_paths:
            result = self.recognize_face(image_path)
            result['image_path'] = image_path
            results.append(result)
        return results

def main():
    # Initialize recognizer
    recognizer = FaceRecognizer()
    
    # Example usage
    test_images = [
        '/Users/aashiqmahato/Downloads/Face detection api/detected_faces/face_0_0_test.jpg'
    ]
    
    # Recognize faces
    for result in recognizer.batch_recognize(test_images):
        print(f"Image: {result['image_path']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Recognized: {result['recognized']}")
            print(f"Name: {result['name']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("---")

if __name__ == '__main__':
    main()
