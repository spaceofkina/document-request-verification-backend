# python-ml/cnn/train_cnn.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split
import json
import time

class PhilippineDocumentCNN:
    def __init__(self):
        self.id_types = [
            'Philippine Passport',
            'UMID (Unified Multi-Purpose ID)',
            'Drivers License (LTO)',
            'Postal ID',
            'National ID (PhilSys)',
            'SSS ID (Social Security System)',
            'Voters ID',
            'PhilHealth ID',
            'Municipal ID',
            'Barangay ID',
            'Student ID'
        ]
        
        self.folder_to_index = {
            'passport': 0,
            'umid': 1,
            'drivers_license': 2,
            'postal_id': 3,
            'national_id': 4,
            'sss_id': 5,
            'voters_id': 6,
            'philhealth_id': 7,
            'municipal_id': 8,
            'barangay_id': 9,
            'student_id': 10
        }
        
        self.model = None
        self.model_accuracy = 0.0
        self.training_stats = {}
        
    def load_images_from_folder(self, base_path):
        """Load images from uploads/real_ids structure"""
        images = []
        labels = []
        image_stats = {}
        
        print("🔍 Scanning Philippine document images...")
        
        # Check primary folder
        primary_path = os.path.join(base_path, 'primary')
        if os.path.exists(primary_path):
            for folder in os.listdir(primary_path):
                if folder in self.folder_to_index:
                    folder_path = os.path.join(primary_path, folder)
                    if os.path.isdir(folder_path):
                        image_files = [f for f in os.listdir(folder_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        
                        if image_files:
                            display_name = self.convert_folder_to_display_name(folder)
                            image_stats[display_name] = len(image_files)
                            print(f"   📂 primary/{folder}: {len(image_files)} images")
                            
                            for img_file in image_files:
                                img_path = os.path.join(folder_path, img_file)
                                images.append(img_path)
                                labels.append(self.folder_to_index[folder])
        
        # Check secondary folder
        secondary_path = os.path.join(base_path, 'secondary')
        if os.path.exists(secondary_path):
            for folder in os.listdir(secondary_path):
                if folder in self.folder_to_index:
                    folder_path = os.path.join(secondary_path, folder)
                    if os.path.isdir(folder_path):
                        image_files = [f for f in os.listdir(folder_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        
                        if image_files:
                            display_name = self.convert_folder_to_display_name(folder)
                            image_stats[display_name] = len(image_files)
                            print(f"   📂 secondary/{folder}: {len(image_files)} images")
                            
                            for img_file in image_files:
                                img_path = os.path.join(folder_path, img_file)
                                images.append(img_path)
                                labels.append(self.folder_to_index[folder])
        
        print(f"\n📊 Found {len(images)} images across {len(image_stats)} document types")
        return images, labels, image_stats
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        return img
    
    def create_cnn_model(self):
        """Create 8-layer CNN similar to your JavaScript version"""
        model = keras.Sequential([
            # Layer 1: Convolutional
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(224, 224, 3),
                         name='conv1_document_classification'),
            
            # Layer 2: Max Pooling
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            # Layer 3: Convolutional
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         name='conv2_feature_extraction'),
            
            # Layer 4: Max Pooling
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            # Layer 5: Flatten
            layers.Flatten(name='flatten_features'),
            
            # Layer 6: Dense
            layers.Dense(128, activation='relu', name='dense1_ph_features'),
            
            # Layer 7: Dropout
            layers.Dropout(0.5, name='dropout_regularization'),
            
            # Layer 8: Output
            layers.Dense(len(self.id_types), activation='softmax', 
                        name='output_ph_document_types')
        ])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ CNN Architecture Created")
        print(f"   Total Layers: 8")
        print(f"   Output: {len(self.id_types)} Philippine document types")
        
        return model
    
    def convert_folder_to_display_name(self, folder_name):
        """Convert folder name to display name"""
        mapping = {
            'passport': 'Philippine Passport',
            'umid': 'UMID (Unified Multi-Purpose ID)',
            'drivers_license': 'Drivers License (LTO)',
            'national_id': 'National ID (PhilSys)',
            'postal_id': 'Postal ID',
            'sss_id': 'SSS ID (Social Security System)',
            'voters_id': 'Voters ID',
            'philhealth_id': 'PhilHealth ID',
            'municipal_id': 'Municipal ID',
            'barangay_id': 'Barangay ID',
            'student_id': 'Student ID'
        }
        
        return mapping.get(folder_name, folder_name.replace('_', ' ').title())
    
    def train(self, data_path='../uploads/real_ids', epochs=10, batch_size=8):
        """Train the CNN model"""
        print("🎓 THESIS: Training CNN for Barangay Document Classification")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load images
        image_paths, labels, image_stats = self.load_images_from_folder(data_path)
        
        if len(image_paths) == 0:
            print("❌ No images found for training")
            return False
        
        print(f"\n🏋️ Training with {len(image_paths)} real Philippine documents...")
        for doc_type, count in image_stats.items():
            print(f"   • {doc_type}: {count} images")
        
        # Preprocess all images
        print("\n📊 Preprocessing images...")
        X = []
        for i, img_path in enumerate(image_paths):
            try:
                img = self.preprocess_image(img_path)
                X.append(img)
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(image_paths)} images")
            except Exception as e:
                print(f"   ✗ Skipped {img_path}: {str(e)}")
                # Remove corresponding label
                labels.pop(i - (len(image_paths) - len(labels)))
        
        X = np.array(X)
        y = np.array(labels)
        
        print(f"✅ Preprocessed {len(X)} images")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create model
        self.model = self.create_cnn_model()
        
        # Train
        print(f"\n📈 Training CNN with {len(X_train)} images...")
        print(f"   Batch size: {batch_size}, Epochs: {epochs}")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Calculate accuracy
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        self.model_accuracy = val_accuracy
        
        # Update stats
        self.training_stats = {
            'totalImages': len(X),
            'documentTypes': len(image_stats),
            'accuracy': float(val_accuracy),
            'realTraining': True,
            'trainingDate': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'epochs': epochs,
            'batchSize': batch_size,
            'imageStats': image_stats,
            'trainingTime': time.time() - start_time
        }
        
        # Save model
        self.save_model()
        
        print(f"\n✅ REAL CNN Training Complete in {time.time() - start_time:.1f} seconds!")
        print(f"   Final Accuracy: {val_accuracy * 100:.1f}%")
        print(f"   Training Images: {len(X)}")
        print(f"   Document Types: {len(image_stats)}")
        
        return True
    
    def save_model(self, save_path='./saved_models'):
        """Save model and training stats"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save TensorFlow model
        model_path = os.path.join(save_path, 'ph_document_cnn')
        self.model.save(model_path)
        
        # Save training stats
        stats_path = os.path.join(save_path, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # Save thesis info
        thesis_info = {
            'thesisTitle': 'Intelligent Document Request Processing System for Barangay Lajong',
            'component': 'Convolutional Neural Network (CNN) for Document Classification',
            'implementation': 'TensorFlow/Keras Python',
            'documentTypes': self.id_types,
            'accuracy': float(self.model_accuracy),
            'trainingImages': self.training_stats.get('totalImages', 0),
            'realTraining': True,
            'framework': 'TensorFlow 2.x',
            'application': 'Barangay Lajong Document Verification',
            'location': 'Bulan, Sorsogon',
            'created': time.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        
        info_path = os.path.join(save_path, 'thesis_info.json')
        with open(info_path, 'w') as f:
            json.dump(thesis_info, f, indent=2)
        
        print(f"💾 Model saved to: {model_path}")
    
    def load_model(self, model_path='./saved_models/ph_document_cnn'):
        """Load trained model"""
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            
            # Load training stats
            stats_path = os.path.join(os.path.dirname(model_path), 'training_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
                    self.model_accuracy = self.training_stats.get('accuracy', 0.0)
            
            print("✅ Loaded pre-trained CNN model")
            return True
        return False
    
    def classify(self, image_path):
        """Classify a single image"""
        if self.model is None:
            print("⚠️ Model not loaded. Loading default...")
            if not self.load_model():
                print("❌ Could not load model")
                return None
        
        try:
            # Preprocess image
            img = self.preprocess_image(image_path)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Predict
            predictions = self.model.predict(img, verbose=0)
            pred_index = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_index])
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions[0])[::-1][:3]
            top_predictions = [
                {
                    'className': self.id_types[idx],
                    'probability': float(predictions[0][idx]),
                    'confidence': float(predictions[0][idx] * 100)
                }
                for idx in top_indices
            ]
            
            result = {
                'detectedIdType': self.id_types[pred_index],
                'confidenceScore': confidence,
                'category': self.get_document_category(self.id_types[pred_index]),
                'isAccepted': True,  # All your document types are accepted
                'allPredictions': top_predictions,
                'processingTime': 0.1,  # Placeholder
                'isRealCNN': True,
                'modelArchitecture': '8-layer CNN (TensorFlow/Keras)',
                'accuracy': float(self.model_accuracy),
                'framework': 'TensorFlow Python'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Classification error: {str(e)}")
            return None
    
    def get_document_category(self, document_type):
        """Categorize as Primary or Secondary"""
        primary_docs = [
            'Philippine Passport', 'UMID', 'Drivers License', 'Postal ID',
            'National ID', 'SSS ID', 'GSIS ID', 'Voters ID', 'PhilHealth ID'
        ]
        
        return 'Primary' if any(doc in document_type for doc in primary_docs) else 'Secondary'

# Main training function
def main():
    cnn = PhilippineDocumentCNN()
    
    # Train with real images
    success = cnn.train(
        data_path='../../uploads/real_ids',  # Relative path from python-ml/cnn/
        epochs=10,
        batch_size=8
    )
    
    if success:
        # Test classification
        print("\n🧪 Testing classification...")
        # Find any image to test
        test_folder = '../../uploads/real_ids/primary/drivers_license'
        if os.path.exists(test_folder):
            test_images = [f for f in os.listdir(test_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if test_images:
                test_path = os.path.join(test_folder, test_images[0])
                result = cnn.classify(test_path)
                if result:
                    print(f"✅ Test Classification:")
                    print(f"   Detected: {result['detectedIdType']}")
                    print(f"   Confidence: {result['confidenceScore']*100:.1f}%")
                    print(f"   Category: {result['category']}")

if __name__ == '__main__':
    main()