import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import cv2

class DataLoader:
    """
    Professional data loader for ASL recognition with:
    - Automated data discovery and loading
    - Feature normalization and scaling
    - Data augmentation for robustness
    - Batch optimization for memory efficiency
    - Quality validation and cleaning
    """
    
    def __init__(self, 
                 train_path: str = "data/train",
                 validation_path: str = "data/validation", 
                 test_path: str = "data/test",
                 sequence_length: int = 30,
                 batch_size: int = 16,
                 validation_split: float = 0.2):
        
        self.train_path = Path(train_path)
        self.validation_path = Path(validation_path) 
        self.test_path = Path(test_path)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Data statistics
        self.data_stats = {}
        
    def discover_data_files(self, data_path: Path) -> List[Path]:
        """Automatically discover all data files in the directory."""
        
        supported_formats = ['.npy', '.npz', '.csv', '.json']
        data_files = []
        
        if not data_path.exists():
            self.logger.warning(f"Data path does not exist: {data_path}")
            return data_files
            
        for file_path in data_path.rglob('*'):
            if file_path.suffix.lower() in supported_formats:
                data_files.append(file_path)
                
        self.logger.info(f"Found {len(data_files)} data files in {data_path}")
        return data_files
    
    def load_numpy_data(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from numpy files with error handling."""
        
        try:
            if file_path.suffix == '.npz':
                data = np.load(file_path, allow_pickle=True)
                # Handle different npz formats
                if 'X' in data and 'y' in data:
                    return data['X'], data['y']
                elif 'features' in data and 'labels' in data:
                    return data['features'], data['labels']
                else:
                    keys = list(data.keys())
                    return data[keys[0]], data[keys[1]]
            else:
                # For .npy files, assume it contains features
                features = np.load(file_path, allow_pickle=True)
                # Extract labels from filename or parent directory
                label = self.extract_label_from_path(file_path)
                labels = np.full(len(features), label)
                return features, labels
                
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return np.array([]), np.array([])
    
    def extract_label_from_path(self, file_path: Path) -> str:
        """Extract label from file path or parent directory."""
        
        # Try parent directory name first
        parent_name = file_path.parent.name
        if parent_name.isalpha() and len(parent_name) == 1:
            return parent_name.upper()
            
        # Try filename
        filename = file_path.stem
        if filename.isalpha() and len(filename) == 1:
            return filename.upper()
            
        # Try extracting from filename patterns
        parts = filename.split('_')
        for part in parts:
            if part.isalpha() and len(part) == 1:
                return part.upper()
                
        # Default fallback
        return 'UNKNOWN'
    
    def validate_sequence_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and clean sequence data."""
        
        if len(features) == 0:
            return features, labels
            
        # Ensure proper shape
        if len(features.shape) == 2:
            # Reshape to sequences if needed
            n_samples = features.shape[0] // self.sequence_length
            if n_samples > 0:
                features = features[:n_samples * self.sequence_length]
                features = features.reshape(n_samples, self.sequence_length, -1)
                labels = labels[:n_samples]
        
        # Remove sequences with NaN or inf values
        valid_mask = np.isfinite(features).all(axis=(1, 2))
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # Remove sequences with all zeros (missing data)
        non_zero_mask = np.any(features != 0, axis=(1, 2))
        features = features[non_zero_mask]
        labels = labels[non_zero_mask]
        
        self.logger.info(f"Cleaned data: {len(features)} valid sequences")
        return features, labels
    
    def augment_data(self, features: np.ndarray, labels: np.ndarray, 
                    augmentation_factor: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to increase dataset size and robustness."""
        
        if augmentation_factor <= 0:
            return features, labels
            
        augmented_features = []
        augmented_labels = []
        
        n_augmented = int(len(features) * augmentation_factor)
        
        for i in range(n_augmented):
            # Random sample
            idx = np.random.randint(0, len(features))
            original_seq = features[idx].copy()
            original_label = labels[idx]
            
            # Apply augmentations
            augmented_seq = self.apply_sequence_augmentation(original_seq)
            
            augmented_features.append(augmented_seq)
            augmented_labels.append(original_label)
        
        if augmented_features:
            augmented_features = np.array(augmented_features)
            augmented_labels = np.array(augmented_labels)
            
            # Combine with original data
            features = np.concatenate([features, augmented_features], axis=0)
            labels = np.concatenate([labels, augmented_labels], axis=0)
            
            self.logger.info(f"Added {len(augmented_features)} augmented samples")
        
        return features, labels
    
    def apply_sequence_augmentation(self, sequence: np.ndarray) -> np.ndarray:
        """Apply various augmentation techniques to a sequence."""
        
        augmented = sequence.copy()
        
        # Gaussian noise (small amount)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented += noise
        
        # Time stretching (slight)
        if np.random.random() < 0.2:
            stretch_factor = np.random.uniform(0.9, 1.1)
            new_length = int(len(augmented) * stretch_factor)
            if new_length > 0:
                indices = np.linspace(0, len(augmented)-1, new_length)
                augmented = np.array([augmented[int(i)] for i in indices])
                
                # Pad or truncate to original length
                if len(augmented) < self.sequence_length:
                    padding = np.zeros((self.sequence_length - len(augmented), augmented.shape[1]))
                    augmented = np.concatenate([augmented, padding], axis=0)
                elif len(augmented) > self.sequence_length:
                    augmented = augmented[:self.sequence_length]
        
        # Small rotation/translation (for pose keypoints)
        if np.random.random() < 0.2:
            # Apply small random transformation
            angle = np.random.uniform(-0.1, 0.1)  # Small rotation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Apply to every 3rd dimension (assuming x,y,z coordinates)
            for i in range(0, augmented.shape[1], 3):
                if i + 1 < augmented.shape[1]:
                    x, y = augmented[:, i], augmented[:, i+1]
                    augmented[:, i] = x * cos_a - y * sin_a
                    augmented[:, i+1] = x * sin_a + y * cos_a
        
        return augmented
    
    def normalize_features(self, features: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Normalize features using StandardScaler."""
        
        if len(features) == 0:
            return features
            
        # Reshape for normalization
        original_shape = features.shape
        features_flat = features.reshape(-1, features.shape[-1])
        
        if fit_scaler:
            normalized = self.scaler.fit_transform(features_flat)
        else:
            normalized = self.scaler.transform(features_flat)
            
        # Reshape back
        normalized = normalized.reshape(original_shape)
        
        return normalized
    
    def encode_labels(self, labels: np.ndarray, fit_encoder: bool = True) -> np.ndarray:
        """Encode string labels to integers."""
        
        if fit_encoder:
            encoded = self.label_encoder.fit_transform(labels)
        else:
            encoded = self.label_encoder.transform(labels)
            
        return encoded
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process training data."""
        
        self.logger.info("Loading training data...")
        
        # Discover data files
        data_files = self.discover_data_files(self.train_path)
        
        all_features = []
        all_labels = []
        
        for file_path in data_files:
            features, labels = self.load_numpy_data(file_path)
            
            if len(features) > 0:
                features, labels = self.validate_sequence_data(features, labels)
                all_features.append(features)
                all_labels.extend(labels)
        
        if all_features:
            features = np.concatenate(all_features, axis=0)
            labels = np.array(all_labels)
            
            # Data augmentation
            features, labels = self.augment_data(features, labels)
            
            # Normalize features
            features = self.normalize_features(features, fit_scaler=True)
            
            # Encode labels
            labels = self.encode_labels(labels, fit_encoder=True)
            
            self.data_stats['train_samples'] = len(features)
            self.data_stats['num_classes'] = len(np.unique(labels))
            self.data_stats['feature_dim'] = features.shape[-1]
            
            self.logger.info(f"Training data loaded: {len(features)} samples, {self.data_stats['num_classes']} classes")
            
            return features, labels
        else:
            self.logger.warning("No training data found!")
            return np.array([]), np.array([])
    
    def load_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process validation data."""
        
        self.logger.info("Loading validation data...")
        
        # Discover data files
        data_files = self.discover_data_files(self.validation_path)
        
        if not data_files:
            self.logger.info("No validation data found, will split from training data")
            return np.array([]), np.array([])
        
        all_features = []
        all_labels = []
        
        for file_path in data_files:
            features, labels = self.load_numpy_data(file_path)
            
            if len(features) > 0:
                features, labels = self.validate_sequence_data(features, labels)
                all_features.append(features)
                all_labels.extend(labels)
        
        if all_features:
            features = np.concatenate(all_features, axis=0)
            labels = np.array(all_labels)
            
            # Normalize features (using fitted scaler)
            features = self.normalize_features(features, fit_scaler=False)
            
            # Encode labels (using fitted encoder)
            labels = self.encode_labels(labels, fit_encoder=False)
            
            self.data_stats['val_samples'] = len(features)
            
            self.logger.info(f"Validation data loaded: {len(features)} samples")
            
            return features, labels
        else:
            return np.array([]), np.array([])
    
    def preprocess_data(self, train_data: Tuple[np.ndarray, np.ndarray], 
                       val_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Final preprocessing including train/val split if needed."""
        
        train_features, train_labels = train_data
        val_features, val_labels = val_data
        
        # If no validation data, split from training
        if len(val_features) == 0 and len(train_features) > 0:
            self.logger.info(f"Splitting training data ({self.validation_split:.1%} for validation)")
            
            train_features, val_features, train_labels, val_labels = train_test_split(
                train_features, train_labels,
                test_size=self.validation_split,
                stratify=train_labels,
                random_state=42
            )
            
            self.data_stats['val_samples'] = len(val_features)
            self.data_stats['train_samples'] = len(train_features)
        
        self.logger.info(f"Final data split: {len(train_features)} train, {len(val_features)} validation")
        
        return (train_features, train_labels), (val_features, val_labels)
    
    def create_batched_dataset(self, data: Tuple[np.ndarray, np.ndarray]) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset with batching."""
        
        features, labels = data
        
        if len(features) == 0:
            return tf.data.Dataset.from_tensor_slices((np.array([]), np.array([])))
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((features.astype(np.float32), labels.astype(np.int32)))
        
        # Shuffle, batch, and prefetch for performance
        dataset = dataset.shuffle(buffer_size=min(1000, len(features)))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_data_statistics(self) -> dict:
        """Return comprehensive data statistics."""
        return self.data_stats.copy()
    
    def save_preprocessing_info(self, save_path: str):
        """Save preprocessing parameters for later use."""
        
        preprocessing_info = {
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'label_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'data_stats': self.data_stats
        }
        
        with open(save_path, 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
            
        self.logger.info(f"Preprocessing info saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # Test the data loader
    data_loader = DataLoader()
    
    train_data = data_loader.load_training_data()
    val_data = data_loader.load_validation_data()
    
    train_data, val_data = data_loader.preprocess_data(train_data, val_data)
    
    train_dataset = data_loader.create_batched_dataset(train_data)
    val_dataset = data_loader.create_batched_dataset(val_data)
    
    print("Data loading complete!")
    print(f"Statistics: {data_loader.get_data_statistics()}") 