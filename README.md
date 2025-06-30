# Advanced CNN-LSTM Sign Language Recognition

> Dissertation Project: Real-time American Sign Language Recognition using Advanced CNN-LSTM Hybrid Architecture

## Project Overview

This project implements a state-of-the-art CNN-LSTM hybrid neural network for real-time American Sign Language (ASL) recognition. The advanced architecture processes MediaPipe keypoints through specialized CNN branches for different body parts (pose, hands, face) before temporal modeling with bidirectional LSTM layers.

### Key Features

- **Advanced CNN-LSTM Hybrid Architecture**: Body-part-aware feature extraction
- **Real-time Performance**: 30+ FPS inference capability  
- **High Accuracy**: Target 70%+ recognition accuracy
- **Comprehensive Preprocessing**: Automated data cleaning and validation
- **Extensive Logging**: Detailed metrics and visualization
- **Batch Training**: Memory-efficient training for large datasets
- **3D Avatar Integration**: Ready for avatar-based sign rendering

## Expected Performance

| Model Architecture | Parameters | Memory | Expected Accuracy |
|-------------------|------------|---------|------------------|
| Basic LSTM | 4.5M | 17.4MB | 10-20% |
| CNN-LSTM Hybrid | 1.5M | 5.6MB | 40-60% |
| **Advanced Hybrid** | **766K** | **2.9MB** | **50-70%** |

## Quick Start

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing
```bash
# Clean and preprocess WLASL dataset
python scripts/preprocess_dataset.py

# Validate processed data
python scripts/validate_data.py
```

### 3. Training
```bash
# Start advanced CNN-LSTM training
python scripts/train_advanced.py

# Monitor training progress
python scripts/monitor_training.py
```

### 4. Real-time Inference
```bash
# Launch real-time recognition
python scripts/realtime_recognition.py
```

## Project Structure

```
├── core/                    # Core implementation
│   ├── models/             # Model architectures
│   ├── preprocessing/      # Data preprocessing
│   ├── training/          # Training utilities
│   ├── evaluation/        # Model evaluation
│   ├── inference/         # Real-time inference
│   └── visualization/     # Plotting utilities
├── data/                   # Dataset files
│   ├── raw/               # Original WLASL data
│   ├── processed/         # Processed keypoints
│   ├── cleaned/           # Final cleaned data
│   └── splits/            # Train/val/test splits
├── models/                 # Saved models
├── results/               # Training results
├── configs/               # Configuration files
├── scripts/               # Automation scripts
└── docs/                  # Documentation
```

## Model Architecture

The Advanced CNN-LSTM Hybrid processes MediaPipe keypoints through:

1. **Body-Part Separation**: 
   - Pose keypoints (132 features)
   - Left/Right hand keypoints (63 each)  
   - Face keypoints (1371 features)

2. **CNN Feature Extraction**:
   - Specialized 1D CNNs for each body part
   - Spatial pattern recognition
   - Feature dimensionality reduction

3. **Temporal Modeling**:
   - Bidirectional LSTM layers
   - Self-attention mechanism
   - Sequence classification

## Training and Evaluation

### Automated Training Pipeline
- **Batch Processing**: Memory-efficient training
- **Comprehensive Logging**: Accuracy, loss, metrics
- **Visualization**: Real-time training plots
- **Checkpointing**: Automatic model saving
- **Early Stopping**: Prevent overfitting

### Performance Monitoring
- Training/validation accuracy curves
- Loss function progression  
- Top-5 accuracy metrics
- Confusion matrices
- Per-class performance analysis

## Configuration

Training parameters can be adjusted in `configs/`:
- `project_config.json`: Main project settings
- `advanced_training.json`: Training hyperparameters
- `data_config.json`: Dataset configuration

## Dissertation Integration

This implementation is structured for dissertation documentation:

### Chapter 5 (Results)
- Comprehensive performance metrics
- Training progression analysis
- Comparative evaluation results
- Visual performance summaries

### Chapter 6 (Discussion)  
- Model architecture justification
- Performance analysis and interpretation
- Comparison with existing literature
- Theoretical and practical implications

## Performance Optimization

- **Mixed Precision Training**: GPU acceleration
- **Data Generators**: Memory-efficient batch loading
- **Caching**: Preprocessed data caching
- **Multiprocessing**: Parallel data loading

## Testing and Validation

```bash
# Run unit tests
python -m pytest tests/

# Validate model performance
python scripts/evaluate_model.py

# Test real-time inference
python scripts/test_realtime.py
```

## Dependencies

See `requirements.txt` for complete dependency list. Key requirements:
- TensorFlow 2.10.0
- MediaPipe 0.10.8
- OpenCV 4.8.0.76
- NumPy 1.24.3

## Contributing

This is a dissertation project. For suggestions or improvements, please create an issue.

## License

This project is for academic research purposes.

---

**Ready to revolutionize sign language recognition!**
