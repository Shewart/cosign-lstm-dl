# Cosign Advanced CNN-LSTM Setup Instructions

## Prerequisites
- Python 3.8-3.11
- Git
- At least 8GB RAM (16GB recommended)
- GPU with CUDA support (optional but recommended)

## Step 1: Clone Repository
```bash
# Replace with your actual repository URL
git clone https://github.com/yourusername/cosine-asl-recognition.git
cd cosine-asl-recognition
```

## Step 2: Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv cosign_env

# Activate virtual environment
# On Windows:
cosign_env\Scripts\activate
# On macOS/Linux:
source cosign_env/bin/activate
```

## Step 3: Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Test GPU availability (if you have one)
python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

## Step 4: Prepare Data Directory
```bash
# Create data directory structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/wlasl

# Copy your WLASL dataset to data/wlasl/
# The structure should be:
# data/wlasl/videos/
# data/wlasl/WLASL_v0.3.json
```

## Step 5: Run Preprocessing
```bash
# Extract keypoints from videos (this will take time)
python scripts/extract_keypoints.py

# Verify preprocessing results
python -c "
import os
processed_dir = 'data/processed'
if os.path.exists(processed_dir):
    files = os.listdir(processed_dir)
    print(f'Processed files: {len(files)}')
    print(f'Sample files: {files[:5]}')
else:
    print('Processed directory not found')
"
```

## Step 6: Test Model Creation
```bash
# Test advanced model creation
python demo_advanced_model.py
```

## Step 7: Start Training
```bash
# Run advanced CNN-LSTM training
python run_advanced_training.py

# Monitor training logs
tail -f logs/advanced_cnn_lstm_training.log
```

## Step 8: Monitor Training Progress
```bash
# Check training results directory
ls -la results/advanced_cnn_lstm_*/

# View training plots (after training starts)
# Training plots will be saved in results/advanced_cnn_lstm_[timestamp]/plots/
```

## Using the Exported Markdown Reference

1. **Save the exported markdown** (`cursor_co-sign.md`) in your project root
2. **Reference it for**:
   - Complete conversation history
   - Detailed implementation explanations
   - Troubleshooting solutions
   - Architecture decisions and rationale

3. **Key sections to reference**:
   - Model architecture explanations
   - Training configuration details
   - Performance optimization tips
   - Error resolution examples

## Quick Commands Summary
```bash
# Complete setup in one go (after cloning)
python -m venv cosign_env
cosign_env\Scripts\activate  # Windows
pip install --upgrade pip
pip install -r requirements.txt
python scripts/extract_keypoints.py
python demo_advanced_model.py
python run_advanced_training.py
```

## Troubleshooting

### If you get MediaPipe errors:
```bash
pip install --upgrade mediapipe
```

### If you get TensorFlow errors:
```bash
pip install tensorflow==2.10.0
```

### If preprocessing fails:
```bash
# Check data directory structure
python -c "
import os
for root, dirs, files in os.walk('data'):
    level = root.replace('data', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files
        print(f'{subindent}{file}')
    if len(files) > 3:
        print(f'{subindent}... and {len(files)-3} more files')
"
```

### If training fails:
```bash
# Check configuration
python -c "
import json
with open('configs/advanced_training_config.json', 'r') as f:
    config = json.load(f)
    print('Batch size:', config['training']['batch_size'])
    print('Learning rate:', config['training']['learning_rate'])
"

# Start with smaller batch size if memory issues
# Edit configs/advanced_training_config.json and reduce batch_size to 8 or 4
```

## Expected Timeline
- **Setup**: 10-15 minutes
- **Preprocessing**: 30-60 minutes (depends on dataset size)
- **Training**: 2-6 hours (depends on hardware and dataset)

## Hardware Recommendations
- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB RAM, GTX 1060/RTX 2060 or better
- **Optimal**: 32GB RAM, RTX 3070/4070 or better

## Next Steps After Setup
1. Monitor training progress in `logs/` directory
2. Check results in `results/advanced_cnn_lstm_*/` directory
3. Use trained model for real-time inference
4. Reference the exported markdown for implementation details 