import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, num_classes):
    """
    Create and return an LSTM model for sign language recognition.
    
    Parameters:
        input_shape (tuple): Shape of the input data (time_steps, features)
        num_classes (int): Number of classification categories

    Returns:
        model (tf.keras.Model): Compiled LSTM model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Example input shape (sequence_length, num_features)
    INPUT_SHAPE = (30, 1662)  # Assuming 1662 features from MediaPipe
    NUM_CLASSES = 10  # Adjust based on actual number of signs

    model = create_lstm_model(INPUT_SHAPE, NUM_CLASSES)
    model.summary()
