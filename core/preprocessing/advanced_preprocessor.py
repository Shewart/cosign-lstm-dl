#!/usr/bin/env python3
"""
Advanced Data Preprocessing Pipeline for Sign Language Recognition
=================================================================

This module implements a sophisticated preprocessing pipeline that:
1. Thoroughly cleans and validates MediaPipe keypoint data
2. Implements advanced filtering and noise reduction
3. Handles missing data and outliers intelligently
4. Provides batch processing for large datasets
5. Generates comprehensive quality reports
6. Optimizes data for CNN-LSTM training

Author: Advanced ASL Recognition Research
Version: 2.0.0
Target: Dissertation-quality preprocessing
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import multiprocessing as mps
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedMediaPipeProcessor:
    """
    Advanced MediaPipe processor with enhanced keypoint extraction and validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MediaPipe processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        
        # Initialize MediaPipe components
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Processing parameters
        self.min_detection_confidence = config.get('min_detection_confidence', 0.7)
        self.min_tracking_confidence = config.get('min_tracking_confidence', 0.5)
        self.static_image_mode = False
        
        # Quality thresholds
        self.min_visibility_threshold = config.get('min_visibility', 0.5)
        self.max_missing_frames_ratio = config.get('max_missing_frames', 0.3)
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=self.static_image_mode,
            model_complexity=2,  # Highest accuracy
            enable_segmentation=False,
            refine_face_landmarks=True,  # Enhanced face landmarks
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        logger.info(f"MediaPipe processor initialized with confidence thresholds: "
                   f"detection={self.min_detection_confidence}, tracking={self.min_tracking_confidence}")
    
    def extract_keypoints_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract comprehensive keypoints from video with quality validation.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Keypoints array of shape (frames, 1629) or None if processing fails
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.error(f"Video has no frames: {video_path}")
            cap.release()
            return None
        
        keypoints_list = []
        quality_metrics = {
            'frames_processed': 0,
            'frames_with_detections': 0,
            'avg_pose_confidence': 0,
            'avg_hand_confidence': 0,
            'avg_face_confidence': 0
        }
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.holistic.process(rgb_frame)
            
            # Extract and validate keypoints
            keypoints, frame_quality = self._extract_frame_keypoints(results)
            
            if keypoints is not None:
                keypoints_list.append(keypoints)
                quality_metrics['frames_with_detections'] += 1
                
                # Update quality metrics
                for metric, value in frame_quality.items():
                    if metric in quality_metrics:
                        quality_metrics[metric] += value
            else:
                # Use previous frame or zeros if first frame
                if keypoints_list:
                    keypoints_list.append(keypoints_list[-1].copy())
                else:
                    keypoints_list.append(np.zeros(1629, dtype=np.float32))
            
            quality_metrics['frames_processed'] += 1
            frame_idx += 1
        
        cap.release()
        
        if not keypoints_list:
            logger.error(f"No valid keypoints extracted from: {video_path}")
            return None
        
        # Convert to numpy array
        keypoints_array = np.array(keypoints_list, dtype=np.float32)
        
        # Validate sequence quality
        if not self._validate_sequence_quality(keypoints_array, quality_metrics):
            logger.warning(f"Low quality sequence detected: {video_path}")
            return None
        
        logger.info(f"Extracted {len(keypoints_list)} frames from {video_path} "
                   f"({quality_metrics['frames_with_detections']}/{quality_metrics['frames_processed']} valid)")
        
        return keypoints_array
    
    def _extract_frame_keypoints(self, results) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """
        Extract comprehensive keypoints from MediaPipe results for a single frame.
        
        Args:
            results: MediaPipe holistic results
            
        Returns:
            Tuple of (keypoints array, quality metrics)
        """
        keypoints = []
        quality_metrics = {
            'avg_pose_confidence': 0.0,
            'avg_hand_confidence': 0.0,
            'avg_face_confidence': 0.0
        }
        
        # Pose landmarks (33 points × 4 features = 132)
        if results.pose_landmarks:
            pose_points = []
            confidences = []
            for landmark in results.pose_landmarks.landmark:
                pose_points.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                confidences.append(landmark.visibility)
            
            keypoints.extend(pose_points)
            quality_metrics['avg_pose_confidence'] = np.mean(confidences)
        else:
            keypoints.extend([0.0] * 132)
        
        # Left hand landmarks (21 points × 3 features = 63)
        if results.left_hand_landmarks:
            left_hand_points = []
            for landmark in results.left_hand_landmarks.landmark:
                left_hand_points.extend([landmark.x, landmark.y, landmark.z])
            
            keypoints.extend(left_hand_points)
            quality_metrics['avg_hand_confidence'] += 0.5  # Assume good quality if detected
        else:
            keypoints.extend([0.0] * 63)
        
        # Right hand landmarks (21 points × 3 features = 63)
        if results.right_hand_landmarks:
            right_hand_points = []
            for landmark in results.right_hand_landmarks.landmark:
                right_hand_points.extend([landmark.x, landmark.y, landmark.z])
            
            keypoints.extend(right_hand_points)
            quality_metrics['avg_hand_confidence'] += 0.5
        else:
            keypoints.extend([0.0] * 63)
        
        # Face landmarks (468 points × 3 features = 1404)
        if results.face_landmarks:
            face_points = []
            for landmark in results.face_landmarks.landmark:
                face_points.extend([landmark.x, landmark.y, landmark.z])
            
            keypoints.extend(face_points)
            quality_metrics['avg_face_confidence'] = 1.0  # Assume good quality if detected
        else:
            keypoints.extend([0.0] * 1404)
        
        # Validate keypoints length
        if len(keypoints) != 1629:
            logger.warning(f"Unexpected keypoints length: {len(keypoints)} (expected 1629)")
            return None, quality_metrics
        
        keypoints_array = np.array(keypoints, dtype=np.float32)
        
        # Check for invalid values
        if np.any(np.isnan(keypoints_array)) or np.any(np.isinf(keypoints_array)):
            logger.warning("NaN or Inf values detected in keypoints")
            return None, quality_metrics
        
        return keypoints_array, quality_metrics
    
    def _validate_sequence_quality(self, keypoints: np.ndarray, 
                                  quality_metrics: Dict[str, float]) -> bool:
        """
        Validate the overall quality of an extracted keypoint sequence.
        
        Args:
            keypoints: Keypoints array of shape (frames, 1629)
            quality_metrics: Quality metrics from extraction
            
        Returns:
            True if sequence meets quality standards
        """
        if len(keypoints) == 0:
            return False
        
        # Check detection rate
        detection_rate = quality_metrics['frames_with_detections'] / quality_metrics['frames_processed']
        if detection_rate < (1 - self.max_missing_frames_ratio):
            logger.warning(f"Low detection rate: {detection_rate:.2f}")
            return False
        
        # Check for excessive zeros (indicating poor detection)
        zero_ratio = np.mean(keypoints == 0)
        if zero_ratio > 0.7:  # More than 70% zeros
            logger.warning(f"High zero ratio: {zero_ratio:.2f}")
            return False
        
        # Check sequence length
        if len(keypoints) < 10:  # Minimum 10 frames
            logger.warning(f"Sequence too short: {len(keypoints)} frames")
            return False
        
        return True
    
    def close(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


class AdvancedDataCleaner:
    """
    Advanced data cleaning and filtering pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data cleaner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Filtering parameters
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        self.smoothing_window = config.get('smoothing_window', 5)
        self.interpolation_method = config.get('interpolation_method', 'linear')
        
        logger.info("Advanced data cleaner initialized")
    
    def clean_sequence(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive cleaning to a keypoint sequence.
        
        Args:
            keypoints: Raw keypoints array of shape (frames, features)
            
        Returns:
            Cleaned keypoints array
        """
        if len(keypoints) == 0:
            return keypoints
        
        cleaned = keypoints.copy()
        
        # 1. Handle missing values (zeros)
        cleaned = self._interpolate_missing_values(cleaned)
        
        # 2. Remove outliers
        cleaned = self._remove_outliers(cleaned)
        
        # 3. Apply smoothing filter
        cleaned = self._apply_smoothing(cleaned)
        
        # 4. Normalize coordinates
        cleaned = self._normalize_coordinates(cleaned)
        
        # 5. Final validation
        cleaned = self._final_validation(cleaned)
        
        return cleaned
    
    def _interpolate_missing_values(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Interpolate missing values (zeros) in keypoint sequences.
        
        Args:
            keypoints: Input keypoints array
            
        Returns:
            Keypoints with interpolated missing values
        """
        if len(keypoints) < 2:
            return keypoints
        
        interpolated = keypoints.copy()
        
        for feature_idx in range(keypoints.shape[1]):
            feature_data = keypoints[:, feature_idx]
            
            # Find non-zero values for interpolation
            non_zero_mask = feature_data != 0
            
            if np.sum(non_zero_mask) < 2:
                continue  # Need at least 2 points for interpolation
            
            # Get indices of non-zero values
            non_zero_indices = np.where(non_zero_mask)[0]
            non_zero_values = feature_data[non_zero_mask]
            
            # Create interpolation function
            if len(non_zero_indices) >= 2:
                interp_func = interp1d(
                    non_zero_indices, 
                    non_zero_values,
                    kind=self.interpolation_method,
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                
                # Interpolate all indices
                all_indices = np.arange(len(feature_data))
                interpolated[:, feature_idx] = interp_func(all_indices)
        
        return interpolated
    
    def _remove_outliers(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Remove outliers using modified Z-score method.
        
        Args:
            keypoints: Input keypoints array
            
        Returns:
            Keypoints with outliers removed
        """
        cleaned = keypoints.copy()
        
        # Calculate median and MAD for each feature
        median = np.median(cleaned, axis=0)
        mad = np.median(np.abs(cleaned - median), axis=0)
        
        # Modified Z-score
        modified_z_scores = 0.6745 * (cleaned - median) / (mad + 1e-8)
        
        # Mark outliers
        outlier_mask = np.abs(modified_z_scores) > self.outlier_threshold
        
        # Replace outliers with median values
        cleaned[outlier_mask] = median[np.newaxis, :]
        
        outlier_count = np.sum(outlier_mask)
        if outlier_count > 0:
            logger.debug(f"Removed {outlier_count} outliers")
        
        return cleaned
    
    def _apply_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to reduce noise.
        
        Args:
            keypoints: Input keypoints array
            
        Returns:
            Smoothed keypoints array
        """
        if len(keypoints) < self.smoothing_window:
            return keypoints
        
        smoothed = keypoints.copy()
        
        # Apply Savitzky-Golay filter for smoothing
        window_length = min(self.smoothing_window, len(keypoints))
        if window_length % 2 == 0:
            window_length -= 1  # Must be odd
        
        if window_length >= 3:
            for feature_idx in range(keypoints.shape[1]):
                try:
                    smoothed[:, feature_idx] = signal.savgol_filter(
                        keypoints[:, feature_idx],
                        window_length=window_length,
                        polyorder=2
                    )
                except:
                    # Fallback to moving average
                    smoothed[:, feature_idx] = np.convolve(
                        keypoints[:, feature_idx],
                        np.ones(window_length) / window_length,
                        mode='same'
                    )
        
        return smoothed
    
    def _normalize_coordinates(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to standard ranges.
        
        Args:
            keypoints: Input keypoints array
            
        Returns:
            Normalized keypoints array
        """
        normalized = keypoints.copy()
        
        # Body part indices based on MediaPipe format
        body_parts = {
            'pose': (0, 132),      # Pose: 33 points × 4 features
            'left_hand': (132, 195),   # Left hand: 21 points × 3 features
            'right_hand': (195, 258),  # Right hand: 21 points × 3 features
            'face': (258, 1629)    # Face: 468 points × 3 features (1371 features)
        }
        
        for part_name, (start_idx, end_idx) in body_parts.items():
            part_data = normalized[:, start_idx:end_idx]
            
            if part_data.shape[1] == 0:
                continue
            
            # Reshape for coordinate processing
            if part_name == 'pose':
                # Pose has x,y,z,visibility format
                coords_per_point = 4
            else:
                # Other parts have x,y,z format
                coords_per_point = 3
            
            num_points = (end_idx - start_idx) // coords_per_point
            
            if num_points > 0:
                reshaped_data = part_data.reshape(-1, num_points, coords_per_point)
                
                # Normalize x,y coordinates to [0,1] range
                for coord_idx in range(min(2, coords_per_point)):  # x, y coordinates
                    coord_data = reshaped_data[:, :, coord_idx]
                    
                    # Remove zeros for normalization calculation
                    non_zero_mask = coord_data != 0
                    if np.any(non_zero_mask):
                        min_val = np.min(coord_data[non_zero_mask])
                        max_val = np.max(coord_data[non_zero_mask])
                        
                        if max_val > min_val:
                            # Normalize non-zero values
                            coord_data[non_zero_mask] = (
                                (coord_data[non_zero_mask] - min_val) / (max_val - min_val)
                            )
                            reshaped_data[:, :, coord_idx] = coord_data
                
                # Reshape back
                normalized[:, start_idx:end_idx] = reshaped_data.reshape(-1, end_idx - start_idx)
        
        return normalized
    
    def _final_validation(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Final validation and cleaning of keypoints.
        
        Args:
            keypoints: Input keypoints array
            
        Returns:
            Final validated keypoints array
        """
        validated = keypoints.copy()
        
        # Replace any remaining NaN or Inf values
        nan_mask = np.isnan(validated) | np.isinf(validated)
        validated[nan_mask] = 0.0
        
        # Clip extreme values
        validated = np.clip(validated, -5.0, 5.0)
        
        return validated.astype(np.float32)


class AdvancedDatasetProcessor:
    """
    Main dataset processing class that orchestrates the entire pipeline.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the dataset processor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.mediapipe_processor = AdvancedMediaPipeProcessor(self.config)
        self.data_cleaner = AdvancedDataCleaner(self.config)
        
        # Processing parameters
        self.batch_size = self.config.get('batch_size', 10)
        self.num_workers = min(self.config.get('num_workers', 4), mps.cpu_count())
        
        # Paths
        self.raw_data_path = Path(self.config['data']['raw_path'])
        self.processed_path = Path(self.config['data']['processed_path'])
        self.cleaned_path = Path(self.config['data']['cleaned_path'])
        
        # Create output directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.cleaned_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset processor initialized with {self.num_workers} workers")
    
    def process_dataset(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process the complete dataset with comprehensive quality control.
        
        Args:
            force_reprocess: Whether to reprocess existing files
            
        Returns:
            Processing report with statistics and quality metrics
        """
        logger.info("🚀 Starting Advanced Dataset Processing")
        
        # Find all video files
        video_files = self._find_video_files()
        logger.info(f"Found {len(video_files)} video files to process")
        
        if not video_files:
            logger.error("No video files found!")
            return {"status": "error", "message": "No video files found"}
        
        # Process videos in batches
        processing_report = {
            "total_videos": len(video_files),
            "processed_successfully": 0,
            "processing_errors": 0,
            "low_quality_sequences": 0,
            "average_sequence_length": 0,
            "quality_metrics": {},
            "error_files": []
        }
        
        # Process in batches to manage memory
        for batch_start in tqdm(range(0, len(video_files), self.batch_size), 
                               desc="Processing batches"):
            batch_end = min(batch_start + self.batch_size, len(video_files))
            batch_files = video_files[batch_start:batch_end]
            
            batch_report = self._process_batch(batch_files, force_reprocess)
            
            # Update overall report
            processing_report["processed_successfully"] += batch_report["successful"]
            processing_report["processing_errors"] += batch_report["errors"]
            processing_report["low_quality_sequences"] += batch_report["low_quality"]
            processing_report["error_files"].extend(batch_report["error_files"])
        
        # Calculate final statistics
        if processing_report["processed_successfully"] > 0:
            processing_report["success_rate"] = (
                processing_report["processed_successfully"] / processing_report["total_videos"]
            )
        else:
            processing_report["success_rate"] = 0.0
        
        # Save processing report
        report_path = self.cleaned_path / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(processing_report, f, indent=2)
        
        logger.info(f"✅ Dataset processing complete!")
        logger.info(f"   Successfully processed: {processing_report['processed_successfully']}")
        logger.info(f"   Errors: {processing_report['processing_errors']}")
        logger.info(f"   Success rate: {processing_report['success_rate']:.2%}")
        
        return processing_report
    
    def _find_video_files(self) -> List[Path]:
        """Find all video files in the raw data directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.raw_data_path.rglob(f"*{ext}"))
        
        return sorted(video_files)
    
    def _process_batch(self, video_files: List[Path], 
                      force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a batch of video files.
        
        Args:
            video_files: List of video file paths
            force_reprocess: Whether to reprocess existing files
            
        Returns:
            Batch processing report
        """
        batch_report = {
            "successful": 0,
            "errors": 0,
            "low_quality": 0,
            "error_files": []
        }
        
        for video_path in video_files:
            try:
                # Check if already processed
                output_name = video_path.stem + ".npy"
                output_path = self.cleaned_path / output_name
                
                if output_path.exists() and not force_reprocess:
                    batch_report["successful"] += 1
                    continue
                
                # Extract keypoints
                keypoints = self.mediapipe_processor.extract_keypoints_from_video(str(video_path))
                
                if keypoints is None:
                    batch_report["errors"] += 1
                    batch_report["error_files"].append(str(video_path))
                    continue
                
                # Clean keypoints
                cleaned_keypoints = self.data_cleaner.clean_sequence(keypoints)
                
                # Validate final quality
                if len(cleaned_keypoints) < 10:  # Minimum sequence length
                    batch_report["low_quality"] += 1
                    logger.warning(f"Low quality sequence: {video_path}")
                    continue
                
                # Save processed keypoints
                np.save(output_path, cleaned_keypoints)
                batch_report["successful"] += 1
                
                logger.debug(f"Processed: {video_path} -> {cleaned_keypoints.shape}")
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                batch_report["errors"] += 1
                batch_report["error_files"].append(str(video_path))
        
        return batch_report
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for processed dataset.
        
        Returns:
            Quality analysis report
        """
        logger.info("📊 Generating Quality Report")
        
        # Find all processed files
        processed_files = list(self.cleaned_path.glob("*.npy"))
        
        if not processed_files:
            return {"error": "No processed files found"}
        
        quality_metrics = {
            "total_sequences": len(processed_files),
            "sequence_lengths": [],
            "feature_statistics": {},
            "missing_data_ratio": [],
            "quality_distribution": {"high": 0, "medium": 0, "low": 0}
        }
        
        # Analyze sample of files
        sample_size = min(100, len(processed_files))
        sample_files = np.random.choice(processed_files, sample_size, replace=False)
        
        for file_path in tqdm(sample_files, desc="Analyzing quality"):
            try:
                keypoints = np.load(file_path)
                
                # Basic statistics
                quality_metrics["sequence_lengths"].append(len(keypoints))
                
                # Missing data analysis
                missing_ratio = np.mean(keypoints == 0)
                quality_metrics["missing_data_ratio"].append(missing_ratio)
                
                # Quality classification
                if missing_ratio < 0.1:
                    quality_metrics["quality_distribution"]["high"] += 1
                elif missing_ratio < 0.3:
                    quality_metrics["quality_distribution"]["medium"] += 1
                else:
                    quality_metrics["quality_distribution"]["low"] += 1
                
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
        
        # Calculate summary statistics
        if quality_metrics["sequence_lengths"]:
            quality_metrics["avg_sequence_length"] = np.mean(quality_metrics["sequence_lengths"])
            quality_metrics["min_sequence_length"] = np.min(quality_metrics["sequence_lengths"])
            quality_metrics["max_sequence_length"] = np.max(quality_metrics["sequence_lengths"])
        
        if quality_metrics["missing_data_ratio"]:
            quality_metrics["avg_missing_ratio"] = np.mean(quality_metrics["missing_data_ratio"])
        
        # Save quality report
        report_path = self.cleaned_path / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)
        
        logger.info(f"✅ Quality report saved to {report_path}")
        return quality_metrics
    
    def cleanup(self):
        """Clean up resources"""
        self.mediapipe_processor.close()


def main():
    """Main processing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced data preprocessing for sign language recognition")
    parser.add_argument("--config", type=str, default="configs/project_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--force-reprocess", action="store_true", 
                       help="Force reprocessing of existing files")
    parser.add_argument("--quality-report", action="store_true",
                       help="Generate quality report only")
    
    args = parser.parse_args()
    
    try:
        processor = AdvancedDatasetProcessor(args.config)
        
        if args.quality_report:
            report = processor.generate_quality_report()
            print(f"Quality report generated: {report}")
        else:
            report = processor.process_dataset(force_reprocess=args.force_reprocess)
            quality_report = processor.generate_quality_report()
            
            print("\n" + "="*50)
            print("📈 PROCESSING COMPLETE")
            print("="*50)
            print(f"Success rate: {report['success_rate']:.2%}")
            print(f"Total processed: {report['processed_successfully']}")
            print(f"Errors: {report['processing_errors']}")
            
        processor.cleanup()
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 