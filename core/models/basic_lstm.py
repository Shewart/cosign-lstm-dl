import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input, Attention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2

def create_attention_layer(query, key, value):
    """Create a self-attention layer for better temporal feature learning"""
    attention_output = Attention()([query, key, value])
    return attention_output

def create_lstm_model(input_shape, num_classes, use_attention=True, is_cpu_only=False):
    """
    Create and return an optimized bidirectional LSTM model for sign language recognition.
    
    Parameters:
        input_shape (tuple): Shape of the input data (time_steps, features)
        num_classes (int): Number of classification categories
        use_attention (bool): Whether to use attention mechanism
        is_cpu_only (bool): If True, use a smaller model for CPU training
    
    Returns:
        model (tf.keras.Model): Compiled bidirectional LSTM model
    """
    # For functional API
    inputs = Input(shape=input_shape)
    
    # Adjust layer sizes based on compute capability
    lstm1_units = 128 if is_cpu_only else 256
    lstm2_units = 64 if is_cpu_only else 128
    dense_units = 64 if is_cpu_only else 128
    
    # First Bidirectional LSTM layer with return sequences for attention
    x = Bidirectional(LSTM(lstm1_units, return_sequences=True, 
                          activation='tanh', 
                          recurrent_activation='sigmoid',
                          recurrent_dropout=0.0,
                          kernel_regularizer=l2(1e-5)))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second Bidirectional LSTM layer
    x = Bidirectional(LSTM(lstm2_units, return_sequences=use_attention,
                          activation='tanh',
                          recurrent_activation='sigmoid',
                          recurrent_dropout=0.0,
                          kernel_regularizer=l2(1e-5)))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Add attention mechanism if specified
    if use_attention:
        x = create_attention_layer(x, x, x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        # Add global pooling to reduce dimensions when using attention
        x = GlobalAveragePooling1D()(x)
    
    # Dense layers for classification
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def compile_model(model, learning_rate=1e-4, clipnorm=1.0):
    """
    Compile the model with optimized parameters
    
    Parameters:
        model: The Keras model to compile
        learning_rate: Learning rate for the optimizer
        clipnorm: Gradient clipping norm
    
    Returns:
        Compiled model
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, 
        clipnorm=clipnorm,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Example usage: adjust NUM_CLASSES as needed.
    INPUT_SHAPE = (50, 1629)  # e.g., 50 frames per sequence, 1629 features per frame
    NUM_CLASSES = 10         # change as needed
    
    # Check if GPU is available
    has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    model = create_lstm_model(INPUT_SHAPE, NUM_CLASSES, is_cpu_only=not has_gpu)
    model = compile_model(model)
    model.summary()
