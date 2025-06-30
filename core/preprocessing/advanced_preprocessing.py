"""
🔥 ADVANCED PREPROCESSING PIPELINE 🔥
=====================================
State-of-the-art data preprocessing for sign language recognition:
- MediaPipe keypoint extraction with quality validation
- Advanced normalization and filtering
- Data augmentation for robustness
- Batch processing for memory efficiency
- Quality metrics and validation
"""

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPipeExtractor:
    """
    Advanced MediaPipe keypoint extraction with quality control
    """
    
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configure holistic model for maximum accuracy
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,  # Highest accuracy
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Keypoint dimensions
        self.pose_landmarks = 33
        self.hand_landmarks = 21
        self.face_landmarks = 468
        
        # Feature structure: pose(132) + left_hand(63) + right_hand(63) + face(1404) = 1662
        self.feature_dim = (self.pose_landmarks * 4) + (self.hand_landmarks * 3 * 2) + (self.face_landmarks * 3)
        
        logger.info(f"🔧 MediaPipe Extractor initialized - Feature dimension: {self.feature_dim}")
    
    def extract_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extract MediaPipe keypoints from a frame with quality metrics
        
        Returns:
            keypoints: Flattened array of all keypoints (1662,)
            quality_metrics: Dictionary with quality scores
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.holistic.process(rgb_frame)
        
        # Initialize keypoints array
        keypoints = np.zeros(self.feature_dim)
        quality_metrics = {
            'pose_detected': False,
            'left_hand_detected': False,
            'right_hand_detected': False,
            'face_detected': False,
            'overall_quality': 0.0
        }
        
        idx = 0
        
        # === POSE KEYPOINTS ===
        if results.pose_landmarks:
            quality_metrics['pose_detected'] = True
            for landmark in results.pose_landmarks.landmark:
                keypoints[idx:idx+4] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                idx += 4
        else:
            idx += self.pose_landmarks * 4
        
        # === LEFT HAND KEYPOINTS ===
        if results.left_hand_landmarks:
            quality_metrics['left_hand_detected'] = True
            for landmark in results.left_hand_landmarks.landmark:
                keypoints[idx:idx+3] = [landmark.x, landmark.y, landmark.z]
                idx += 3
        else:
            idx += self.hand_landmarks * 3
        
        # === RIGHT HAND KEYPOINTS ===
        if results.right_hand_landmarks:
            quality_metrics['right_hand_detected'] = True
            for landmark in results.right_hand_landmarks.landmark:
                keypoints[idx:idx+3] = [landmark.x, landmark.y, landmark.z]
                idx += 3
        else:
            idx += self.hand_landmarks * 3
        
        # === FACE KEYPOINTS ===
        if results.face_landmarks:
            quality_metrics['face_detected'] = True
            for landmark in results.face_landmarks.landmark:
                keypoints[idx:idx+3] = [landmark.x, landmark.y, landmark.z]
                idx += 3
        else:
            idx += self.face_landmarks * 3
        
        # Calculate overall quality score
        detected_parts = sum([
            quality_metrics['pose_detected'],
            quality_metrics['left_hand_detected'],
            quality_metrics['right_hand_detected'],
            quality_metrics['face_detected']
        ])
        quality_metrics['overall_quality'] = detected_parts / 4.0
        
        return keypoints, quality_metrics
    
    def process_video(self, video_path: str, max_frames: int = 30) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract keypoints from entire video with quality tracking
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames_data = []
        quality_data = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints, quality = self.extract_keypoints(frame)
            frames_data.append(keypoints)
            quality_data.append(quality)
            frame_count += 1
        
        cap.release()
        
        # Convert to numpy array and pad if necessary
        if len(frames_data) < max_frames:
            # Pad with zeros if video is shorter than max_frames
            padding_needed = max_frames - len(frames_data)
            for _ in range(padding_needed):
                frames_data.append(np.zeros(self.feature_dim))
                quality_data.append({
                    'pose_detected': False,
                    'left_hand_detected': False,
                    'right_hand_detected': False,
                    'face_detected': False,
                    'overall_quality': 0.0
                })
        
        sequence = np.array(frames_data[:max_frames])
        
        logger.info(f"Processed video: {video_path} - Shape: {sequence.shape}")
        return sequence, quality_data


class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline with data cleaning and augmentation
    """
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"🔧 Advanced Preprocessor initialized - Sequence length: {sequence_length}")
    
    def remove_outliers(self, data: np.ndarray, method: str = 'iqr', factor: float = 1.5) -> np.ndarray:
        """
        Remove outliers from keypoint data
        """
        if method == 'iqr':
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Clip outliers
            data_cleaned = np.clip(data, lower_bound, upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            mask = z_scores < factor
            data_cleaned = data.copy()
            data_cleaned[~mask] = np.mean(data, axis=0)
        
        else:
            data_cleaned = data
        
        return data_cleaned
    
    def smooth_sequence(self, sequence: np.ndarray, window_length: int = 5) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to reduce noise
        """
        if len(sequence) < window_length:
            return sequence
        
        smoothed = np.zeros_like(sequence)
        
        for i in range(sequence.shape[1]):
            # Only smooth non-zero sequences (ignore padding)
            non_zero_mask = sequence[:, i] != 0
            if np.sum(non_zero_mask) >= window_length:
                smoothed[non_zero_mask, i] = savgol_filter(
                    sequence[non_zero_mask, i], 
                    window_length=min(window_length, np.sum(non_zero_mask)), 
                    polyorder=2
                )
            else:
                smoothed[:, i] = sequence[:, i]
        
        return smoothed
    
    def interpolate_missing(self, sequence: np.ndarray) -> np.ndarray:
        """
        Interpolate missing keypoints (zeros) using linear interpolation
        """
        interpolated = sequence.copy()
        
        for i in range(sequence.shape[1]):
            # Find non-zero indices
            non_zero_indices = np.where(sequence[:, i] != 0)[0]
            
            if len(non_zero_indices) >= 2:
                # Interpolate zeros between non-zero values
                for j in range(len(sequence)):
                    if sequence[j, i] == 0:
                        # Find nearest non-zero neighbors
                        left_idx = non_zero_indices[non_zero_indices < j]
                        right_idx = non_zero_indices[non_zero_indices > j]
                        
                        if len(left_idx) > 0 and len(right_idx) > 0:
                            left = left_idx[-1]
                            right = right_idx[0]
                            
                            # Linear interpolation
                            weight = (j - left) / (right - left)
                            interpolated[j, i] = (1 - weight) * sequence[left, i] + weight * sequence[right, i]
        
        return interpolated
    
    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize sequence using fitted scaler
        """
        original_shape = sequence.shape
        sequence_flat = sequence.reshape(-1, sequence.shape[-1])
        
        if not self.is_fitted:
            normalized_flat = self.scaler.fit_transform(sequence_flat)
            self.is_fitted = True
            logger.info("✅ Scaler fitted on training data")
        else:
            normalized_flat = self.scaler.transform(sequence_flat)
        
        return normalized_flat.reshape(original_shape)
    
    def augment_sequence(self, sequence: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
        """
        Apply data augmentation techniques
        """
        if np.random.random() > augment_prob:
            return sequence
        
        augmented = sequence.copy()
        
        # Random noise injection (small)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, sequence.shape)
            augmented += noise
        
        # Random time warping (slight)
        if np.random.random() < 0.2:
            # Simple time dilation/compression
            stretch_factor = np.random.uniform(0.9, 1.1)
            original_length = len(sequence)
            new_length = int(original_length * stretch_factor)
            
            if new_length > 0:
                indices = np.linspace(0, original_length - 1, new_length)
                indices = np.round(indices).astype(int)
                augmented = sequence[indices]
                
                # Pad or trim to original length
                if len(augmented) < original_length:
                    padding = np.zeros((original_length - len(augmented), sequence.shape[1]))
                    augmented = np.vstack([augmented, padding])
                else:
                    augmented = augmented[:original_length]
        
        # Random spatial jitter (very small)
        if np.random.random() < 0.2:
            spatial_noise = np.random.normal(0, 0.005, sequence.shape)
            augmented += spatial_noise
        
        return augmented
    
    def process_sequence(self, sequence: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single sequence
        """
        # Step 1: Remove outliers
        processed = self.remove_outliers(sequence)
        
        # Step 2: Interpolate missing values
        processed = self.interpolate_missing(processed)
        
        # Step 3: Smooth the sequence
        processed = self.smooth_sequence(processed)
        
        # Step 4: Apply augmentation (only during training)
        if is_training:
            processed = self.augment_sequence(processed)
        
        # Step 5: Normalize
        processed = self.normalize_sequence(processed)
        
        return processed


class DatasetProcessor:
    """
    Complete dataset processing and validation pipeline
    """
    
    def __init__(self, sequence_length: int = 30, batch_size: int = 32):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        
        self.extractor = MediaPipeExtractor()
        self.preprocessor = AdvancedPreprocessor(sequence_length)
        
        self.dataset_stats = {}
        
        logger.info(f"🔧 Dataset Processor initialized")
    
    def process_dataset(self, 
                       data_dir: Union[str, Path], 
                       output_dir: Union[str, Path],
                       split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Dict:
        """
        Process entire dataset with train/val/test split
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"�� Processing dataset from {data_dir}")
        
        # Discover all video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(list(data_dir.rglob(ext)))
        
        if not video_files:
            raise ValueError(f"No video files found in {data_dir}")
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Extract class labels from directory structure
        class_labels = {}
        label_to_idx = {}
        
        processed_data = []
        quality_reports = []
        
        # Process videos in batches to manage memory
        for i in tqdm(range(0, len(video_files), self.batch_size), desc="Processing videos"):
            batch_files = video_files[i:i + self.batch_size]
            
            for video_path in batch_files:
                try:
                    # Extract class label from parent directory
                    class_name = video_path.parent.name
                    
                    if class_name not in label_to_idx:
                        label_to_idx[class_name] = len(label_to_idx)
                    
                    class_idx = label_to_idx[class_name]
                    
                    # Extract keypoints
                    sequence, quality_data = self.extractor.process_video(str(video_path), self.sequence_length)
                    
                    # Calculate average quality
                    avg_quality = np.mean([q['overall_quality'] for q in quality_data])
                    
                    # Only keep high-quality sequences
                    if avg_quality >= 0.5:  # At least 50% of body parts detected on average
                        processed_data.append({
                            'sequence': sequence,
                            'label': class_idx,
                            'class_name': class_name,
                            'video_path': str(video_path),
                            'quality_score': avg_quality
                        })
                        
                        quality_reports.append({
                            'video_path': str(video_path),
                            'class_name': class_name,
                            'quality_score': avg_quality,
                            'frame_count': len([q for q in quality_data if q['overall_quality'] > 0])
                        })
                    else:
                        logger.warning(f"Low quality video skipped: {video_path} (quality: {avg_quality:.2f})")
                
                except Exception as e:
                    logger.error(f"Error processing {video_path}: {str(e)}")
                    continue
        
        logger.info(f"Successfully processed {len(processed_data)} videos")
        
        # Create train/val/test splits
        np.random.shuffle(processed_data)
        
        train_end = int(len(processed_data) * split_ratios[0])
        val_end = train_end + int(len(processed_data) * split_ratios[1])
        
        train_data = processed_data[:train_end]
        val_data = processed_data[train_end:val_end]
        test_data = processed_data[val_end:]
        
        # Process and save splits
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            if split_data:
                # Extract sequences and labels
                sequences = np.array([item['sequence'] for item in split_data])
                labels = np.array([item['label'] for item in split_data])
                
                # Apply preprocessing
                processed_sequences = np.array([
                    self.preprocessor.process_sequence(seq, is_training=(split_name == 'train'))
                    for seq in tqdm(sequences, desc=f"Preprocessing {split_name}")
                ])
                
                # Save processed data
                np.save(output_dir / f"{split_name}_sequences.npy", processed_sequences)
                np.save(output_dir / f"{split_name}_labels.npy", labels)
                
                logger.info(f"✅ {split_name.title()} set: {len(processed_sequences)} samples")
        
        # Save metadata
        metadata = {
            'num_classes': len(label_to_idx),
            'class_names': {v: k for k, v in label_to_idx.items()},
            'label_to_idx': label_to_idx,
            'sequence_length': self.sequence_length,
            'feature_dim': self.extractor.feature_dim,
            'split_sizes': {
                'train': len(train_data),
                'validation': len(val_data),
                'test': len(test_data)
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save quality report
        quality_df = pd.DataFrame(quality_reports)
        quality_df.to_csv(output_dir / 'quality_report.csv', index=False)
        
        # Generate quality visualization
        self._generate_quality_plots(quality_df, output_dir)
        
        logger.info(f"🎉 Dataset processing complete! Saved to {output_dir}")
        
        return metadata
    
    def _generate_quality_plots(self, quality_df: pd.DataFrame, output_dir: Path):
        """
        Generate quality analysis plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Quality score distribution
        axes[0, 0].hist(quality_df['quality_score'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Quality Score Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Quality by class
        sns.boxplot(data=quality_df, x='class_name', y='quality_score', ax=axes[0, 1])
        axes[0, 1].set_title('Quality Score by Class')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Frame count distribution
        axes[1, 0].hist(quality_df['frame_count'], bins=15, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Frame Count Distribution')
        axes[1, 0].set_xlabel('Valid Frames')
        axes[1, 0].set_ylabel('Frequency')
        
        # Class distribution
        class_counts = quality_df['class_name'].value_counts()
        axes[1, 1].bar(class_counts.index, class_counts.values, color='coral')
        axes[1, 1].set_title('Samples per Class')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Quality analysis plots saved")


def load_processed_dataset(data_dir: Union[str, Path]) -> Tuple[Dict, Dict]:
    """
    Load preprocessed dataset
    """
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load data splits
    data = {}
    for split in ['train', 'validation', 'test']:
        seq_path = data_dir / f"{split}_sequences.npy"
        label_path = data_dir / f"{split}_labels.npy"
        
        if seq_path.exists() and label_path.exists():
            data[split] = {
                'sequences': np.load(seq_path),
                'labels': np.load(label_path)
            }
            logger.info(f"Loaded {split}: {data[split]['sequences'].shape}")
    
    return data, metadata


if __name__ == "__main__":
    # Test preprocessing pipeline
    print("🧪 Testing Advanced Preprocessing Pipeline...")
    
    # Create test processor
    processor = DatasetProcessor(sequence_length=30, batch_size=16)
    
    print("✅ Preprocessing pipeline test successful!") 