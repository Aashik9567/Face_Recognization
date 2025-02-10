import os
import cv2
import torch
import numpy as np
import logging
from facenet_pytorch import InceptionResnetV1
from typing import List, Tuple
import imghdr  # Added to validate image types

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)

class FaceEmbeddingGenerator:
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize FaceNet model for embedding generation
        
        Args:
            model_path (str, optional): Path to pre-trained model weights
            device (str, optional): Computation device (cuda/cpu)
        """
        # Determine device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        try:
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            # If a custom model path is provided, load custom weights
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logging.info(f"Loaded custom weights from {model_path}")
        
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            raise

    def is_image_file(self, filename: str) -> bool:
        """
        Check if the file is a valid image
        
        Args:
            filename (str): Path to the file
        
        Returns:
            bool: True if the file is a valid image, False otherwise
        """
        try:
            # List of valid image extensions
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            
            # Check file extension
            if os.path.splitext(filename)[1].lower() not in valid_extensions:
                return False
            
            # Additional check using imghdr
            image_type = imghdr.what(filename)
            return image_type is not None
        
        except Exception as e:
            logging.warning(f"Error checking image file {filename}: {e}")
            return False

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for embedding generation
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Validate image
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to FaceNet input size
            img = cv2.resize(img, (160, 160))
            
            # Normalize and convert to tensor
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            
            return img.to(self.device)
        
        except Exception as e:
            logging.error(f"Image preprocessing error for {image_path}: {e}")
            return None

    def generate_embeddings(self, known_faces_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate face embeddings for all images in the directory
        
        Args:
            known_faces_dir (str): Directory containing face images
        
        Returns:
            Tuple of numpy arrays: (embeddings, names)
        """
        known_encodings = []
        known_names = []

        # Validate input directory
        if not os.path.exists(known_faces_dir):
            raise ValueError(f"Directory not found: {known_faces_dir}")

        # Iterate through person folders
        for person_name in os.listdir(known_faces_dir):
            person_folder = os.path.join(known_faces_dir, person_name)
            
            # Skip if not a directory
            if not os.path.isdir(person_folder):
                continue
            
            logging.info(f"Processing images for {person_name}...")
            
            # Process each image in person's folder
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                
                # Skip non-image files
                if not self.is_image_file(image_path):
                    logging.info(f"Skipping non-image file: {image_path}")
                    continue
                
                # Preprocess image
                img_tensor = self.preprocess_image(image_path)
                
                if img_tensor is None:
                    continue
                
                # Generate embedding
                try:
                    with torch.no_grad():
                        embedding = self.model(img_tensor).squeeze().cpu().numpy()
                    
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    known_encodings.append(embedding)
                    known_names.append(person_name)
                
                except Exception as e:
                    logging.error(f"Embedding generation error for {image_path}: {e}")

        # Convert to numpy arrays
        known_encodings = np.array(known_encodings)
        known_names = np.array(known_names)

        return known_encodings, known_names

    def save_embeddings(self, 
                         encodings: np.ndarray, 
                         names: np.ndarray, 
                         encoding_path: str = 'known_encodings.npy',
                         names_path: str = 'known_names.npy'):
        """
        Save generated embeddings to files
        
        Args:
            encodings (np.ndarray): Face embeddings
            names (np.ndarray): Corresponding names
            encoding_path (str): Path to save encodings
            names_path (str): Path to save names
        """
        try:
            np.save(encoding_path, encodings)
            np.save(names_path, names)
            logging.info(f"Saved {len(encodings)} embeddings")
        except Exception as e:
            logging.error(f"Embedding saving error: {e}")

def main():
    try:
        # Paths
        model_path = "20180402-114759-vggface2.pt"  # Optional custom model path
        known_faces_dir = "processed_faces"  # Directory containing known faces
        
        # Initialize generator
        generator = FaceEmbeddingGenerator(model_path)
        
        # Generate embeddings
        encodings, names = generator.generate_embeddings(known_faces_dir)
        
        # Save embeddings
        generator.save_embeddings(encodings, names)
    
    except Exception as e:
        logging.error(f"Main process error: {e}")

if __name__ == '__main__':
    main()
