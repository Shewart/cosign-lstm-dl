"""
data_preprocessing_fixed.py
--------------------
Utility functions for extracting and processing keypoints from MediaPipe results
and preparing them for use with the LSTM model.

This is an optimized version with better memory management and error handling.
"""

import numpy as np
import cv2
import os
import json
import tensorflow as tf
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe results into a flattened vector
    
    Args:
        results: MediaPipe Holistic model results
        
    Returns:
        np.ndarray: Flattened vector of keypoints (1629 features)
    """
    # Initialize empty arrays
    pose = np.zeros(33 * 3)  # 33 pose landmarks * 3 values (x, y, z)
    face = np.zeros(468 * 3)  # 468 face landmarks * 3 values
    lh = np.zeros(21 * 3)     # 21 left hand landmarks * 3 values
    rh = np.zeros(21 * 3)     # 21 right hand landmarks * 3 values
    
    # If landmarks exist, convert them to numpy arrays
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
    
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    
    # Concatenate all features
    return np.concatenate([pose, face, lh, rh])

def normalize_keypoints(keypoints):
    """
    Normalize keypoints using Z-score standardization
    
    Args:
        keypoints: numpy array of keypoints
        
    Returns:
        np.ndarray: Normalized keypoints
    """
    # Skip normalization if all zeros (empty frame)
    if np.all(keypoints == 0):
        return keypoints
        
    # Z-score standardization (mean=0, std=1)
    mean = np.mean(keypoints)
    std = np.std(keypoints)
    if std > 0:
        return (keypoints - mean) / std
    return keypoints

def preprocess_video(video_path, output_dir, extraction_fps=15, sequence_length=None, max_frames=300):
    """
    Extract keypoints from a video file and save as .npy
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save processed keypoints
        extraction_fps: FPS to extract frames at (lower = faster processing)
        sequence_length: Number of frames to extract (None for all frames)
        max_frames: Maximum number of frames to process (to prevent memory issues)
        
    Returns:
        str: Path to saved .npy file or None if failed
    """
    # Log the start of processing
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video {video_id}...")
    
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{video_id}.npy")
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skipping {video_id} - already processed")
            return output_path
            
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval based on desired extraction FPS
        frame_interval = max(1, int(fps / extraction_fps))
        
        # Limit frames to process
        if max_frames and (total_frames // frame_interval) > max_frames:
            print(f"Limiting {video_id} to {max_frames} frames (original: {total_frames})")
            max_frames_to_process = max_frames * frame_interval
        else:
            max_frames_to_process = total_frames
        
        # Initialize MediaPipe with more memory-efficient settings
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,  # Lower complexity = faster but less accurate
            smooth_landmarks=True,  # Smoothing helps reduce jitter
            refine_face_landmarks=False  # Don't need detailed face landmarks
        )
        
        # Extract keypoints
        keypoints_sequence = []
        current_frame = 0
        frames_processed = 0
        
        try:
            while cap.isOpened() and current_frame < max_frames_to_process and (sequence_length is None or len(keypoints_sequence) < sequence_length):
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames based on interval
                if current_frame % frame_interval != 0:
                    current_frame += 1
                    continue
                
                # Resize frame to reduce memory usage
                frame = cv2.resize(frame, (320, 240))
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Set writeable flag to False for performance
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)
                frame_rgb.flags.writeable = True
                
                # Extract and normalize keypoints
                keypoints = extract_keypoints(results)
                keypoints = normalize_keypoints(keypoints)
                
                # Add to sequence
                keypoints_sequence.append(keypoints)
                
                current_frame += 1
                frames_processed += 1
                
                # Log progress every 50 frames
                if frames_processed % 50 == 0:
                    print(f"  {video_id}: Processed {frames_processed} frames")
                
                # Clear memory periodically
                if frames_processed % 100 == 0:
                    # Delete frame variables to help with garbage collection
                    del frame, frame_rgb, results
                    gc.collect()
        
        except Exception as e:
            print(f"Error during frame processing for {video_id}: {e}")
            # Continue with what we have if we've processed at least some frames
            if len(keypoints_sequence) < 5:  # Arbitrary minimum threshold
                return None
        
        finally:
            # Release resources
            cap.release()
            holistic.close()
        
        # If no keypoints extracted, return None
        if not keypoints_sequence:
            print(f"Warning: No keypoints extracted from {video_path}")
            return None
            
        # Save as numpy array
        keypoints_array = np.array(keypoints_sequence)
        np.save(output_path, keypoints_array)
        
        print(f"Successfully saved {video_id} with {len(keypoints_sequence)} frames")
        return output_path
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Ensure garbage collection runs
        gc.collect()

def batch_process_videos(video_dir, output_dir, json_mapping=None, num_workers=4):
    """
    Process multiple videos in parallel
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save processed keypoints
        json_mapping: Optional path to JSON mapping file
        num_workers: Number of parallel workers
        
    Returns:
        list: Paths to saved .npy files
    """
    # Get list of video files
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                  if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return []
        
    # Load JSON mapping if provided
    video_id_whitelist = None
    if json_mapping and os.path.exists(json_mapping):
        try:
            with open(json_mapping, 'r') as f:
                json_data = json.load(f)
                
            # Extract video IDs that have gloss labels
            video_id_whitelist = set()
            for entry in json_data:
                for instance in entry.get('instances', []):
                    video_id = str(instance.get('video_id')).zfill(5)
                    video_id_whitelist.add(video_id)
                    
            print(f"Loaded {len(video_id_whitelist)} video IDs from JSON mapping")
        except Exception as e:
            print(f"Error loading JSON mapping: {e}")
    
    # Filter videos based on whitelist if needed
    if video_id_whitelist:
        filtered_videos = []
        for video_path in video_files:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            if video_id in video_id_whitelist:
                filtered_videos.append(video_path)
        
        print(f"Filtered {len(filtered_videos)} videos out of {len(video_files)} based on JSON mapping")
        video_files = filtered_videos
    
    # Skip videos that are already processed
    existing_files = set(os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.npy'))
    video_files_to_process = []
    for video_path in video_files:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        if video_id not in existing_files:
            video_files_to_process.append(video_path)
    
    print(f"Skipping {len(video_files) - len(video_files_to_process)} already processed videos")
    print(f"Processing {len(video_files_to_process)} videos")
    
    if not video_files_to_process:
        print("No new videos to process")
        return []
    
    # Process videos in smaller batches of parallel tasks
    BATCH_SIZE = 50  # Process 50 videos at a time to avoid memory issues
    processed_files = []
    
    for i in range(0, len(video_files_to_process), BATCH_SIZE):
        batch = video_files_to_process[i:i+BATCH_SIZE]
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(len(video_files_to_process) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        # Process videos in parallel
        batch_processed = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(preprocess_video, video_path, output_dir): video_path
                for video_path in batch
            }
            
            # Process as they complete
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    result = future.result()
                    if result:
                        batch_processed.append(result)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
        
        processed_files.extend(batch_processed)
        print(f"Batch complete - processed {len(batch_processed)}/{len(batch)} videos")
        
        # Force garbage collection between batches
        gc.collect()
        
        # Pause between batches to allow resources to be freed
        if i + BATCH_SIZE < len(video_files_to_process):
            print("Taking a short break...")
            import time
            time.sleep(5)
    
    return processed_files

def generate_train_val_split(data_dir, json_path, val_split=0.2, save_to_file=True):
    """
    Generate train/validation split for processed keypoints
    
    Args:
        data_dir: Directory containing processed .npy files
        json_path: Path to JSON mapping file
        val_split: Fraction of data to use for validation
        save_to_file: Whether to save split to file
        
    Returns:
        tuple: (train_files, val_files)
    """
    # Get all processed files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if not all_files:
        print(f"No .npy files found in {data_dir}")
        return [], []
        
    # Load JSON mapping
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON mapping: {e}")
        return [], []
        
    # Create video ID to gloss mapping
    videoid_to_gloss = {}
    for entry in json_data:
        gloss = entry.get("gloss")
        for instance in entry.get("instances", []):
            video_id = str(instance.get("video_id")).zfill(5)
            videoid_to_gloss[video_id] = gloss
    
    # Create file list with labels
    file_list = []
    for f in all_files:
        video_id = os.path.splitext(f)[0]
        if video_id in videoid_to_gloss:
            gloss = videoid_to_gloss[video_id]
            file_list.append((f, gloss))
    
    # Shuffle files
    np.random.shuffle(file_list)
    
    # Split into train/val
    split_idx = int(len(file_list) * (1 - val_split))
    train_files = file_list[:split_idx]
    val_files = file_list[split_idx:]
    
    print(f"Split {len(file_list)} files into {len(train_files)} train and {len(val_files)} validation")
    
    # Save to file if requested
    if save_to_file:
        with open(os.path.join(data_dir, 'train_files.json'), 'w') as f:
            json.dump(train_files, f)
        with open(os.path.join(data_dir, 'val_files.json'), 'w') as f:
            json.dump(val_files, f)
    
    return train_files, val_files

def main():
    """Process command line arguments and execute batch processing"""
    parser = argparse.ArgumentParser(description='Process sign language videos to extract keypoints')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed keypoints')
    parser.add_argument('--json_mapping', type=str, default=None,
                        help='Path to JSON mapping file (optional)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of parallel workers (default: 2)')
    parser.add_argument('--extraction_fps', type=int, default=15,
                        help='FPS to extract frames at (default: 15)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting video processing with {args.num_workers} workers")
    print(f"Reading videos from: {args.video_dir}")
    print(f"Saving processed data to: {args.output_dir}")
    
    processed_files = batch_process_videos(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        json_mapping=args.json_mapping,
        num_workers=args.num_workers
    )
    
    print(f"✅ Processing complete. Processed {len(processed_files)} videos.")
    
    # If JSON mapping provided, generate train/val split
    if args.json_mapping and os.path.exists(args.json_mapping):
        train_files, val_files = generate_train_val_split(
            data_dir=args.output_dir,
            json_path=args.json_mapping,
            val_split=0.2,
            save_to_file=True
        )
        print(f"✅ Generated train/val split: {len(train_files)} train, {len(val_files)} validation")

if __name__ == "__main__":
    main() 