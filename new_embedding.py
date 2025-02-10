import os
import numpy as np
import torch
import logging
import cv2
from facenet_pytorch import InceptionResnetV1
import imghdr

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)

class EmbeddingUpdater:
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

    def generate_person_embeddings(self, person_folder: str) -> list:
        """
        Generate embeddings for a single person's images
        
        Args:
            person_folder (str): Folder containing images of the person
        
        Returns:
            list: Embeddings for the person
        """
        person_embeddings = []

        # Validate input directory
        if not os.path.exists(person_folder):
            raise ValueError(f"Directory not found: {person_folder}")

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
                
                person_embeddings.append(embedding)
            
            except Exception as e:
                logging.error(f"Embedding generation error for {image_path}: {e}")

        return person_embeddings

    def update_embeddings(self, 
                           new_person_name: str, 
                           new_person_folder: str, 
                           encoding_path: str = 'known_encodings.npy',
                           names_path: str = 'known_names.npy'):
        """
        Update existing embedding files with new person's embeddings
        
        Args:
            new_person_name (str): Name of the new person
            new_person_folder (str): Folder containing new person's images
            encoding_path (str): Path to existing encodings file
            names_path (str): Path to existing names file
        """
        try:
            # Load existing embeddings and names
            existing_encodings = np.load(encoding_path)
            existing_names = np.load(names_path)

            # Generate embeddings for the new person
            new_person_embeddings = self.generate_person_embeddings(new_person_folder)

            if not new_person_embeddings:
                logging.warning(f"No valid embeddings found for {new_person_name}")
                return

            # Combine existing and new embeddings
            updated_encodings = np.vstack([existing_encodings, np.array(new_person_embeddings)])
            
            # Create corresponding names array
            updated_names = np.concatenate([
                existing_names, 
                np.full(len(new_person_embeddings), new_person_name)
            ])

            # Save updated embeddings
            np.save(encoding_path, updated_encodings)
            np.save(names_path, updated_names)

            logging.info(f"Added {len(new_person_embeddings)} embeddings for {new_person_name}")
            logging.info(f"Total embeddings now: {len(updated_encodings)}")

        except FileNotFoundError:
            # If existing files don't exist, create new ones
            encodings = np.array(self.generate_person_embeddings(new_person_folder))
            names = np.full(len(encodings), new_person_name)

            np.save(encoding_path, encodings)
            np.save(names_path, names)

            logging.info(f"Created new embedding files with {len(encodings)} embeddings for {new_person_name}")
        
        except Exception as e:
            logging.error(f"Embedding update error: {e}")

def main():
    try:
        # Paths
        model_path = "20180402-114759-vggface2.pt"  # Optional custom model path
        new_person_name = "NewPerson"  # Replace with actual name
        new_person_folder = "cropped_faces/NewPerson"  # Replace with actual folder path
        
        # Initialize updater
        updater = EmbeddingUpdater(model_path)
        
        # Update embeddings
        updater.update_embeddings(new_person_name, new_person_folder)
    
    except Exception as e:
        logging.error(f"Main process error: {e}")

if __name__ == '__main__':
    main()