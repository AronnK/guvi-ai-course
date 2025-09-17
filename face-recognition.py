import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from mtcnn.mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
import requests
import tarfile
import random
import pickle
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Install required packages
# !pip install tensorflow mtcnn scikit-learn tensorflow_hub

class FaceRecognitionSystem:
    def __init__(self):
        """Initializes the face recognition system components."""
        self.detector = MTCNN()
        # Load EfficientNet model directly using KerasLayer wrapper
        self.facenet_model_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2", trainable=False)
        self.required_size = (160, 160)
        self.embeddings = []
        self.labels = []
        self.le = LabelEncoder()
        self.classifier = SVC(kernel='linear', C=0.1, probability=True)

    def _extract_face(self, image):
        """Extracts a single face from an image."""
        faces = self.detector.detect_faces(image)
        if not faces:
            return None
        
        x1, y1, width, height = faces[0]['box']
        x2, y2 = x1 + width, y1 + height
        
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, self.required_size)
        return face

    def _get_embedding(self, face):
        """Gets the embedding for a face image."""
        # Preprocess the face image
        face = face.astype('float32') / 255.0 # Normalize pixel values
        
        # Add batch dimension and get embedding
        face_tensor = tf.convert_to_tensor([face])
        embedding = self.facenet_model_layer(face_tensor)
        return embedding.numpy()[0]

    def process_dataset(self, dataset_path, max_samples_per_person=20):
        """Processes the dataset to extract face embeddings and labels."""
        person_samples = {}
        # First, find eligible people (those with enough samples)
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:max_samples_per_person]
                
                # Only include people with at least 5 images
                if len(images) >= 5:
                    person_samples[person_name] = images

        print(f"Found {len(person_samples)} people with sufficient samples")

        # Process faces for each eligible person
        for person_name, images in tqdm(person_samples.items(), desc="Processing people"):
            successful_embeddings = 0
            for image_name in images:
                try:
                    image_path = os.path.join(dataset_path, person_name, image_name)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face = self._extract_face(image)
                    
                    if face is not None:
                        embedding = self._get_embedding(face)
                        self.embeddings.append(embedding)
                        self.labels.append(person_name)
                        successful_embeddings += 1
                except Exception as e:
                    print(f"Error processing image {image_path}: {str(e)}")
            
            if successful_embeddings > 0:
                print(f"Processed {successful_embeddings} images for {person_name}")

    def train_classifier(self):
        """Trains the SVM classifier on the extracted embeddings."""
        if len(self.embeddings) == 0:
            print("No embeddings to train on!")
            return None
        
        X = np.array(self.embeddings)
        encoded_labels = self.le.fit_transform(self.labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_labels, test_size=0.2, random_state=42
        )
        
        self.classifier.fit(X_train, y_train)
        
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
        }

def plot_results(results):
    """Plots the training data distribution and model accuracy."""
    if results is None:
        print("No results to plot!")
        return
        
    plt.figure(figsize=(15, 5))
    
    # Plot training data distribution
    plt.subplot(1, 2, 1)
    plt.title('Training Data Distribution')
    unique, counts = np.unique(results['y_train'], return_counts=True)
    plt.bar(unique[:10], counts[:10]) # Plot only first 10 classes for visibility
    plt.xlabel('Person ID (first 10)')
    plt.ylabel('Number of samples')
    
    # Plot confusion matrix and accuracy
    plt.subplot(1, 2, 2)
    plt.title(f"Model Accuracy: {results['accuracy']:.2%}")
    cm = tf.math.confusion_matrix(results['y_test'], results['y_pred'])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    plt.tight_layout()
    plt.show()

# Main execution block
if __name__ == "__main__":
    try:
        # Initialize system and process dataset
        print("Initializing Face Recognition System...")
        face_system = FaceRecognitionSystem()
        
        print("Processing dataset...")
        dataset_path = '/content/lfw'
        face_system.process_dataset(dataset_path, max_samples_per_person=20)
        
        # Train classifier and get results
        print(f"Training classifier on {len(face_system.embeddings)} samples...")
        results = face_system.train_classifier()
        
        if results is not None:
            # Plot and display results
            plot_results(results)
            
            # Print classification report
            y_true = face_system.le.inverse_transform(results['y_test'])
            y_pred = face_system.le.inverse_transform(results['y_pred'])
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
            
            # Save model
            save_path = '/content/drive/MyDrive/face_recognition_model.pkl'
            if os.path.exists('/content/drive'):
                import pickle
                model_data = {
                    'classifier': face_system.classifier,
                    'label_encoder': face_system.le
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(model_data, f)
                print(f"Model saved to Google Drive: {save_path}")
            else:
                print("Training failed due to lack of directory.")
    except Exception as e:
        print(f"An error occurred: {e}")