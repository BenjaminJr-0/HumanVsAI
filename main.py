import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AIImageDetector:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN architecture for binary classification"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def create_data_generators(self, train_dir, validation_split=0.2):
        """Create data generators with augmentation"""
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50):
        """Train the model with callbacks"""
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Visualize training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_generator):
        """Evaluate model and create confusion matrix"""
        # Predictions
        predictions = self.model.predict(test_generator)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Real', 'AI-Generated']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'AI-Generated'],
                   yticklabels=['Real', 'AI-Generated'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm, y_pred, y_true
    
    def predict_single_image(self, img_path):
        """Predict if a single image is AI-generated"""
        img = keras.preprocessing.image.load_img(
            img_path, target_size=(self.img_height, self.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = self.model.predict(img_array)[0][0]
        
        # Display result
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        
        if prediction > 0.5:
            label = f"AI-Generated ({prediction*100:.2f}% confidence)"
            color = 'red'
        else:
            label = f"Real Image ({(1-prediction)*100:.2f}% confidence)"
            color = 'green'
        
        plt.title(label, fontsize=14, color=color, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return prediction
    
    def save_model(self, filepath='ai_detector_model.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='ai_detector_model.h5'):
        """Load pretrained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("AI-Generated Image Detector")
    print("="*50)
    
    # Initialize detector
    detector = AIImageDetector(img_height=224, img_width=224)
    
    # Build model
    model = detector.build_model()
    print("\nModel Architecture:")
    model.summary()
    
    # Note: You need to organize your dataset like this:
    # data/
    #   ├── real/
    #   │   ├── img1.jpg
    #   │   ├── img2.jpg
    #   └── ai_generated/
    #       ├── img1.jpg
    #       ├── img2.jpg
    
    # Uncomment below to train (requires organized dataset)
    """
    # Create data generators
    train_gen, val_gen = detector.create_data_generators('data/', validation_split=0.2)
    
    # Train model
    print("\nStarting training...")
    detector.train(train_gen, val_gen, epochs=50)
    
    # Plot training history
    detector.plot_training_history()
    
    # Evaluate on validation set
    detector.evaluate_model(val_gen)
    
    # Save model
    detector.save_model('ai_detector_model.h5')
    
    # Predict single image
    prediction = detector.predict_single_image('path/to/test/image.jpg')
    """
    
    print("\nSetup complete! Organize your dataset and uncomment training code to start.")
    print(tf.__version__)