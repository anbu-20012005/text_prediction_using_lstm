import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def train_model(x,y,vocab_size,max_len):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_len - 1,input_shape=(max_len-1,)))
    model.add(LSTM(150,return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    print("🚀 Training started...")
    model.fit(x, y, epochs=100, verbose=1)
    print("✅ Training complete!")

     # Make sure model directory exists
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)
    # Save the trained model
    model_path = os.path.join(model_dir, "text_predictor.h5")
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")

    return model
