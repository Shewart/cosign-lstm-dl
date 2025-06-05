#!/usr/bin/env python3
"""
🔥 STEP 4: REAL-TIME INFERENCE 🔥
=================================
Real-time sign language recognition inference
"""

import sys
import os
from pathlib import Path
import json
import cv2
import numpy as np

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.preprocessing.advanced_preprocessing import MediaPipeExtractor, AdvancedPreprocessor
import tensorflow as tf
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignLanguageInference:
    """
    Real-time sign language recognition inference
    """
    
    def __init__(self, model_path: str, metadata_path: str):
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = [self.metadata['class_names'][str(i)] 
                           for i in range(self.metadata['num_classes'])]
        self.sequence_length = self.metadata['sequence_length']
        
        # Initialize preprocessors
        self.extractor = MediaPipeExtractor()
        self.preprocessor = AdvancedPreprocessor(self.sequence_length)
        
        # Sequence buffer for real-time processing
        self.sequence_buffer = []
        
        logger.info(f"✅ Inference system initialized")
        logger.info(f"   📊 Classes: {len(self.class_names)}")
        logger.info(f"   📊 Sequence Length: {self.sequence_length}")
    
    def predict_video(self, video_path: str) -> dict:
        """
        Predict sign language from video file
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints
            keypoints, quality = self.extractor.extract_keypoints(frame)
            frames.append(keypoints)
            frame_count += 1
        
        cap.release()
        
        # Pad sequence if necessary
        while len(frames) < self.sequence_length:
            frames.append(np.zeros(self.extractor.feature_dim))
        
        # Convert to numpy array and preprocess
        sequence = np.array(frames[:self.sequence_length])
        processed_sequence = self.preprocessor.process_sequence(sequence, is_training=False)
        
        # Add batch dimension
        batch_sequence = np.expand_dims(processed_sequence, axis=0)
        
        # Make prediction
        predictions = self.model.predict(batch_sequence, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {self.class_names[i]: float(predictions[0][i]) 
                              for i in range(len(self.class_names))}
        }
    
    def predict_webcam(self, show_video: bool = True):
        """
        Real-time prediction from webcam
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        print("🚀 Starting real-time inference... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints
            keypoints, quality = self.extractor.extract_keypoints(frame)
            
            # Add to sequence buffer
            self.sequence_buffer.append(keypoints)
            
            # Keep only last sequence_length frames
            if len(self.sequence_buffer) > self.sequence_length:
                self.sequence_buffer.pop(0)
            
            # Make prediction when buffer is full
            if len(self.sequence_buffer) == self.sequence_length:
                # Preprocess sequence
                sequence = np.array(self.sequence_buffer)
                processed_sequence = self.preprocessor.process_sequence(sequence, is_training=False)
                batch_sequence = np.expand_dims(processed_sequence, axis=0)
                
                # Predict
                predictions = self.model.predict(batch_sequence, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                # Display results
                if show_video:
                    # Add prediction text to frame
                    text = f"{self.class_names[predicted_class]}: {confidence:.2f}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    
                    # Add quality indicator
                    quality_text = f"Quality: {quality['overall_quality']:.2f}"
                    cv2.putText(frame, quality_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 255), 2)
                
                print(f"Prediction: {self.class_names[predicted_class]} "
                      f"(Confidence: {confidence:.3f})")
            
            if show_video:
                cv2.imshow('Sign Language Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Sign Language Recognition Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.keras)')
    parser.add_argument('--metadata_path', type=str, required=True,
                       help='Path to metadata.json file')
    parser.add_argument('--mode', type=str, choices=['video', 'webcam'], 
                       default='webcam', help='Inference mode')
    parser.add_argument('--video_path', type=str, 
                       help='Path to video file (required for video mode)')
    parser.add_argument('--show_video', action='store_true',
                       help='Show video output')
    
    args = parser.parse_args()
    
    print("🔥" * 50)
    print("🚀 SIGN LANGUAGE RECOGNITION INFERENCE")
    print("🔥" * 50)
    print(f"🤖 Model: {args.model_path}")
    print(f"📋 Metadata: {args.metadata_path}")
    print(f"🎯 Mode: {args.mode}")
    print("🔥" * 50)
    
    try:
        # Initialize inference system
        inference = SignLanguageInference(args.model_path, args.metadata_path)
        
        if args.mode == 'video':
            if not args.video_path:
                logger.error("❌ Video path required for video mode")
                return 1
            
            print(f"🎬 Processing video: {args.video_path}")
            result = inference.predict_video(args.video_path)
            
            print("\n🎉 PREDICTION RESULTS:")
            print(f"✅ Predicted Sign: {result['predicted_class']}")
            print(f"✅ Confidence: {result['confidence']:.3f}")
            
        elif args.mode == 'webcam':
            inference.predict_webcam(show_video=args.show_video)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 