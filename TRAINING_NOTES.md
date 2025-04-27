# Training Optimization Notes

## Overview of Improvements

This document outlines the improvements made to the sign language recognition training pipeline to address several issues:

1. **Negative Percentages in Training Logs** - Fixed by adding proper clamping to metric values
2. **High Loss Values** - Addressed through improved normalization and regularization
3. **Model Saving Errors** - Implemented robust saving with fallbacks
4. **Training Stability** - Enhanced with better data generators and optimization settings

## Key Changes in `train_optimized.py`

### Data Processing Improvements

- **Robust Normalization**: Added epsilon value (`1e-3`) to prevent division by zero
- **NaN Handling**: Added `np.nan_to_num()` to replace NaN or infinity values in sequences
- **Improved Padding**: More explicit handling of sequence length variations
- **Expanded Caching**: Increased cache size for better performance

### Training Stability Fixes

- **Batch Size**: Increased to 32 for better gradient estimates
- **Learning Rate**: Set to fixed `3e-4` value with proper clipping
- **Gradient Clipping**: Added `clipnorm=1.0` to prevent exploding gradients
- **Epsilon Adjustment**: Increased optimizer epsilon to `1e-7` for numerical stability

### Metrics and Logging

- **Percentage Fix**: Properly clamped accuracy values to [0,1] range before percentage calculation
- **Pretty Logging**: Added better formatted output with headers
- **Progress Information**: Enhanced dataset statistics reporting

### Model Saving Improvements

- **Staged Saving**: Primary save with fallback to weights-only if full save fails
- **Checkpoint Separation**: Better separation between checkpoints and final model
- **Error Handling**: Comprehensive try/except blocks around saving code
- **Interruption Handling**: Added keyboard interrupt catching with safe model saving

## Training Parameters

The optimal settings found for training are:

```python
# Training parameters
BATCH_SIZE = 32        # Larger batch for stable gradients
EPOCHS = 100           # Train longer with early stopping
VALIDATION_SPLIT = 0.2 # 80/20 split
LEARNING_RATE = 3e-4   # Adjusted learning rate
DROPOUT_RATE = 0.5     # Increased dropout for better generalization
L2_REGULARIZATION = 1e-5  # Weight regularization to prevent overfitting
CLIP_NORM = 1.0        # Gradient clipping to prevent exploding gradients
```

## How to Use the New Training System

1. Run the main training script from the project root:
   ```
   python train.py
   ```

2. This will automatically use the optimized training script in `src/train_optimized.py`

3. For MacBook Air M1 compatibility, the script automatically disables GPU requirements

4. Training progress is displayed with fixed percentages and properly clamped values

5. Model checkpoints are saved to `models/lstm_model_checkpoint.h5` during training

6. The final model is saved to `models/lstm_model.h5` at completion

## Debugging

If training still shows issues, check:

1. **Data Quality** - Ensure `processed_vids/wlasl-video/processed_keypoints_fixed` contains properly formatted `.npy` files
2. **Memory Usage** - If running out of memory, reduce `BATCH_SIZE` to 16 or 8
3. **Learning Rate** - If loss is unstable, try reducing to `1e-4` or `5e-5`
4. **Early Stopping** - If the model plateaus quickly, try increasing `patience` in early stopping callback 