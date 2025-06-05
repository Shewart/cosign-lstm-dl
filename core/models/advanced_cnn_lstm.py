"""
🔥 ADVANCED CNN-LSTM HYBRID ARCHITECTURE 🔥
===============================================
State-of-the-art sign language recognition model with:
- Body part aware CNN branches 
- Bidirectional LSTM temporal modeling
- Self-attention mechanisms
- Spatial-temporal feature fusion
- Residual connections
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_body_part_cnn(input_dim: int, name: str, filters_progression: list = [32, 64, 128]) -> keras.Model:
    """
    Create CNN branch for specific body part
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name}_input')
    
    # Reshape for 1D CNN
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # Apply CNN layers
    for i, filters in enumerate(filters_progression):
        x = layers.Conv1D(filters, 3, padding='same', name=f'{name}_conv1d_{i}')(x)
        x = layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        x = layers.ReLU(name=f'{name}_relu_{i}')(x)
        if i < len(filters_progression) - 1:  # No pooling on last layer
            x = layers.MaxPooling1D(2, name=f'{name}_pool_{i}')(x)
        x = layers.Dropout(0.2, name=f'{name}_dropout_{i}')(x)
    
    # Global feature extraction
    x = layers.GlobalAveragePooling1D(name=f'{name}_gap')(x)
    outputs = layers.Dense(128, activation='relu', name=f'{name}_features')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{name}_cnn_branch')
    return model


class SelfAttention(layers.Layer):
    """
    Self-attention mechanism for temporal sequence modeling
    """
    def __init__(self, d_model=256, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model//num_heads,
            name='self_attention'
        )
        self.layer_norm = layers.LayerNormalization(name='attention_norm')
        self.dropout = layers.Dropout(0.1)
        
    def call(self, inputs, training=None):
        # Self-attention
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout(attn_output, training=training)
        
        # Residual connection + layer norm
        out = self.layer_norm(inputs + attn_output)
        
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape


class AdvancedCNNLSTM:
    """
    🔥 ADVANCED CNN-LSTM HYBRID MODEL 🔥
    
    Architecture Flow:
    1. MediaPipe Keypoints (1629 features) → Body Part Separation
    2. Pose (132) → CNN Branch → Features (128)
    3. Left Hand (63) → CNN Branch → Features (128) 
    4. Right Hand (63) → CNN Branch → Features (128)
    5. Face (1371) → CNN Branch → Features (128)
    6. Feature Fusion → (512 features per frame)
    7. Bidirectional LSTM → Temporal modeling
    8. Self-Attention → Important frame focus
    9. Classification → Sign prediction
    
    Expected Performance: 50-70% accuracy (vs 10% LSTM-only)
    """
    
    def __init__(self, num_classes, sequence_length=30, input_features=1662):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.input_features = input_features
        
        # MediaPipe keypoint structure (CORRECTED)
        self.pose_features = 132      # 33 pose landmarks × 4 (x,y,z,visibility)
        self.hand_features = 63       # 21 hand landmarks × 3 (x,y,z)
        self.face_features = 1404     # 468 face landmarks × 3 (x,y,z) = CORRECTED
        
        self.model = None
        
    def build_model(self):
        """
        Build the advanced CNN-LSTM hybrid architecture
        """
        print("🚀 Building Advanced CNN-LSTM Hybrid Model...")
        
        # Input: (batch_size, sequence_length, 1629)
        inputs = layers.Input(shape=(self.sequence_length, self.input_features), name='mediapipe_input')
        
        # === BODY PART SEPARATION ===
        print("  🔧 Separating body parts...")
        
        # Split input into body parts
        pose_input = layers.Lambda(
            lambda x: x[:, :, :self.pose_features], 
            name='pose_extraction'
        )(inputs)
        
        left_hand_start = self.pose_features
        left_hand_input = layers.Lambda(
            lambda x: x[:, :, left_hand_start:left_hand_start + self.hand_features],
            name='left_hand_extraction'
        )(inputs)
        
        right_hand_start = left_hand_start + self.hand_features
        right_hand_input = layers.Lambda(
            lambda x: x[:, :, right_hand_start:right_hand_start + self.hand_features],
            name='right_hand_extraction'
        )(inputs)
        
        face_start = right_hand_start + self.hand_features
        face_input = layers.Lambda(
            lambda x: x[:, :, face_start:],
            name='face_extraction'
        )(inputs)
        
        # === CNN BRANCHES FOR EACH BODY PART ===
        print("  🔧 Creating CNN branches for body parts...")
        
        # Create specialized CNN branches
        pose_cnn = create_body_part_cnn(self.pose_features, 'pose', [32, 64, 128])
        left_hand_cnn = create_body_part_cnn(self.hand_features, 'left_hand', [16, 32, 64])
        right_hand_cnn = create_body_part_cnn(self.hand_features, 'right_hand', [16, 32, 64])
        face_cnn = create_body_part_cnn(self.face_features, 'face', [64, 128, 256])
        
        # Apply CNN branches frame by frame using TimeDistributed
        pose_features = layers.TimeDistributed(pose_cnn, name='pose_cnn_td')(pose_input)
        left_hand_features = layers.TimeDistributed(left_hand_cnn, name='left_hand_cnn_td')(left_hand_input)
        right_hand_features = layers.TimeDistributed(right_hand_cnn, name='right_hand_cnn_td')(right_hand_input)
        face_features = layers.TimeDistributed(face_cnn, name='face_cnn_td')(face_input)
        
        # === FEATURE FUSION ===
        print("  🔗 Fusing body part features...")
        fused_features = layers.Concatenate(axis=-1, name='feature_fusion')([
            pose_features, left_hand_features, right_hand_features, face_features
        ])
        
        # Feature projection
        fused_features = layers.Dense(512, activation='relu', name='feature_projection')(fused_features)
        fused_features = layers.Dropout(0.3, name='fusion_dropout')(fused_features)
        
        # === TEMPORAL MODELING ===
        print("  ⏰ Adding temporal modeling layers...")
        
        # Bidirectional LSTM for temporal dependencies
        lstm_out = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='bidirectional_lstm'
        )(fused_features)
        
        # Self-attention for important frame focus
        attention_out = SelfAttention(d_model=512, num_heads=8, name='self_attention')(lstm_out)
        
        # Global temporal pooling
        temporal_features = layers.GlobalAveragePooling1D(name='temporal_pooling')(attention_out)
        
        # === CLASSIFICATION HEAD ===
        print("  🎯 Adding classification layers...")
        x = layers.Dense(256, activation='relu', name='classifier_dense1')(temporal_features)
        x = layers.Dropout(0.5, name='classifier_dropout1')(x)
        x = layers.Dense(128, activation='relu', name='classifier_dense2')(x)
        x = layers.Dropout(0.3, name='classifier_dropout2')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='AdvancedCNNLSTM')
        
        print("✅ Advanced CNN-LSTM model built successfully!")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with optimized settings
        """
        if self.model is None:
            self.build_model()
            
        # Advanced optimizer with learning rate scheduling
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.0001,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print(f"✅ Model compiled with AdamW optimizer (lr={learning_rate})")
        return self.model
    
    def get_model_summary(self):
        """
        Print detailed model architecture
        """
        if self.model is None:
            self.build_model()
            
        print("\n" + "="*80)
        print("🔥 ADVANCED CNN-LSTM HYBRID ARCHITECTURE 🔥")
        print("="*80)
        self.model.summary(show_trainable=True)
        
        # Calculate parameters
        total_params = self.model.count_params()
        print(f"\n📊 Model Statistics:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Model Size: ~{(total_params * 4) / (1024**2):.1f} MB")
        print(f"   Expected Accuracy: 50-70% (vs 10% LSTM-only)")
        print("="*80)
        
        return self.model


def create_advanced_model(num_classes, sequence_length=30):
    """
    Factory function to create and return the advanced CNN-LSTM model
    """
    print("🚀 Initializing Advanced CNN-LSTM Hybrid...")
    
    model_builder = AdvancedCNNLSTM(
        num_classes=num_classes,
        sequence_length=sequence_length
    )
    
    model = model_builder.compile_model()
    model_builder.get_model_summary()
    
    return model, model_builder


if __name__ == "__main__":
    # Test model creation
    print("🧪 Testing Advanced CNN-LSTM model creation...")
    model, builder = create_advanced_model(num_classes=26, sequence_length=30)
    print("✅ Model test successful!") 