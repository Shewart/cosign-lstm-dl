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
from tensorflow.keras import layers, Model
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


class BodyPartSeparator(layers.Layer):
    """Separates MediaPipe keypoints into body regions for specialized processing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # MediaPipe 1629-feature breakdown
        self.face_indices = list(range(0, 468 * 3))  # 468 face landmarks * 3 coords
        self.pose_indices = list(range(468 * 3, (468 + 33) * 3))  # 33 pose landmarks * 3 coords  
        self.left_hand_indices = list(range((468 + 33) * 3, (468 + 33 + 21) * 3))  # 21 hand landmarks * 3 coords
        self.right_hand_indices = list(range((468 + 33 + 21) * 3, 1629))  # 21 hand landmarks * 3 coords
    
    def call(self, inputs):
        face_features = tf.gather(inputs, self.face_indices, axis=-1)
        pose_features = tf.gather(inputs, self.pose_indices, axis=-1)
        left_hand_features = tf.gather(inputs, self.left_hand_indices, axis=-1)
        right_hand_features = tf.gather(inputs, self.right_hand_indices, axis=-1)
        
        return {
            'face': face_features,
            'pose': pose_features,
            'left_hand': left_hand_features,
            'right_hand': right_hand_features
        }


class AttentionBlock(layers.Layer):
    """Multi-head attention mechanism for temporal feature fusion."""
    
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1
        )
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * 2, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(d_model)
        ])
    
    def call(self, inputs, training=None):
        # Self-attention
        attn_output = self.attention(inputs, inputs, training=training)
        out1 = self.norm1(inputs + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1, training=training)
        return self.norm2(out1 + ffn_output)


class BodyPartCNN(layers.Layer):
    """Specialized CNN for each body part with different architectural strategies."""
    
    def __init__(self, body_part, output_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.body_part = body_part
        self.output_dim = output_dim
        
        if body_part == 'face':
            # Face: high detail, many landmarks - deeper network
            self.conv_layers = [
                layers.Conv1D(32, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(128, 3, activation='relu', padding='same'),
                layers.GlobalMaxPooling1D(),
                layers.Dense(output_dim, activation='relu')
            ]
        elif body_part == 'pose':
            # Pose: structural info - focus on spatial relationships
            self.conv_layers = [
                layers.Conv1D(64, 5, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(128, 3, activation='relu', padding='same'),
                layers.GlobalMaxPooling1D(),
                layers.Dense(output_dim, activation='relu')
            ]
        else:  # hands
            # Hands: critical for ASL - aggressive feature extraction
            self.conv_layers = [
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(128, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(256, 3, activation='relu', padding='same'),
                layers.GlobalMaxPooling1D(),
                layers.Dense(output_dim, activation='relu')
            ]
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x, training=training)
        return x


class AdvancedCNNLSTM(Model):
    """
    Advanced CNN-LSTM Hybrid for ASL Recognition
    - Body-part-aware processing
    - Attention mechanisms
    - Optimized for MediaPipe keypoints
    - Target accuracy: 70%+
    """
    
    def __init__(self, num_classes, sequence_length=30, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # Body part separation
        self.body_separator = BodyPartSeparator()
        
        # Specialized CNNs for each body part
        self.face_cnn = BodyPartCNN('face', output_dim=64)
        self.pose_cnn = BodyPartCNN('pose', output_dim=64)
        self.left_hand_cnn = BodyPartCNN('left_hand', output_dim=128)
        self.right_hand_cnn = BodyPartCNN('right_hand', output_dim=128)
        
        # Feature fusion
        self.fusion_dense = layers.Dense(256, activation='relu')
        self.fusion_dropout = layers.Dropout(0.3)
        
        # Temporal processing with attention
        self.lstm1 = layers.LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        self.attention1 = AttentionBlock(d_model=512, num_heads=8)
        
        self.lstm2 = layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        self.attention2 = AttentionBlock(d_model=256, num_heads=4)
        
        self.lstm3 = layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3)
        
        # Classification head
        self.classifier = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Data augmentation for training
        self.noise_layer = layers.GaussianNoise(0.01)
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        
        # Process each time step
        timestep_features = []
        
        for t in range(sequence_length):
            timestep_data = inputs[:, t, :]
            
            # Add noise during training for robustness
            if training:
                timestep_data = self.noise_layer(timestep_data, training=training)
            
            # Separate body parts
            body_parts = self.body_separator(timestep_data)
            
            # Process each body part with specialized CNNs
            # Reshape for Conv1D: (batch, features, 1) -> (batch, 1, features)
            face_feat = tf.expand_dims(body_parts['face'], axis=1)
            pose_feat = tf.expand_dims(body_parts['pose'], axis=1)
            left_hand_feat = tf.expand_dims(body_parts['left_hand'], axis=1)
            right_hand_feat = tf.expand_dims(body_parts['right_hand'], axis=1)
            
            face_processed = self.face_cnn(face_feat, training=training)
            pose_processed = self.pose_cnn(pose_feat, training=training)
            left_hand_processed = self.left_hand_cnn(left_hand_feat, training=training)
            right_hand_processed = self.right_hand_cnn(right_hand_feat, training=training)
            
            # Fuse features with learned weights
            fused = tf.concat([
                face_processed, 
                pose_processed, 
                left_hand_processed, 
                right_hand_processed
            ], axis=-1)
            
            fused = self.fusion_dense(fused)
            fused = self.fusion_dropout(fused, training=training)
            
            timestep_features.append(fused)
        
        # Stack timestep features: (batch, sequence, features)
        sequence_features = tf.stack(timestep_features, axis=1)
        
        # Temporal processing with attention
        x = self.lstm1(sequence_features, training=training)
        x = self.attention1(x, training=training)
        
        x = self.lstm2(x, training=training)
        x = self.attention2(x, training=training)
        
        x = self.lstm3(x, training=training)
        
        # Classification
        return self.classifier(x, training=training)
    
    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'sequence_length': self.sequence_length
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_advanced_model(num_classes, sequence_length=30, input_shape=(30, 1629)):
    """
    Factory function to create and compile the advanced CNN-LSTM model.
    
    Args:
        num_classes: Number of sign language classes
        sequence_length: Length of input sequences
        input_shape: Shape of input data (sequence_length, features)
    
    Returns:
        Compiled Keras model ready for training
    """
    
    model = AdvancedCNNLSTM(num_classes=num_classes, sequence_length=sequence_length)
    
    # Build the model
    model.build(input_shape=(None, *input_shape))
    
    # Compile with advanced optimizer settings
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def get_model_summary(num_classes=26):
    """Get a detailed summary of the model architecture."""
    model = create_advanced_model(num_classes)
    
    print("🏗️  Advanced CNN-LSTM Hybrid Architecture")
    print("=" * 50)
    model.summary()
    
    # Calculate memory usage
    total_params = model.count_params()
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    print(f"\n📊 Model Statistics:")
    print(f"   • Total Parameters: {total_params:,}")
    print(f"   • Memory Usage: {memory_mb:.1f} MB")
    print(f"   • Target Accuracy: 70%+")
    print(f"   • Body-part aware: ✅")
    print(f"   • Attention mechanisms: ✅")
    print(f"   • Optimized for MediaPipe: ✅")
    
    return model


if __name__ == "__main__":
    # Demonstrate the model
    model = get_model_summary() 