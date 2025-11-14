import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def load_dataset():
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "..", "data", "dataset.txt")
    data_path = os.path.abspath(data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    return text_data

def tokenization(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    return tokenizer

def input_creation(text,tokenizer):
    input_sequences = []
    for sentence in text.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1,len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i+1])
    return input_sequences

def padding(input_sequences):
    max_len = max([len(x) for x in input_sequences])
    padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')
    return padded_input_sequences,max_len

def create_labels(padded_input_sequences,tokenizer):
    x = padded_input_sequences[:,:-1]
    y = padded_input_sequences[:,-1]
    return x,y

def save_tokenizer(tokenizer):
    current_dir = os.path.dirname(__file__)
    tokenizer_dir = os.path.join(current_dir, "..", "model")
    os.makedirs(tokenizer_dir, exist_ok=True)

    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_json(), f, ensure_ascii=False, indent=2)

    print(f"💾 Tokenizer saved to: {tokenizer_path}")

def preprocess():
    text = load_dataset()
    tokenizer = tokenization(text)
    input_sequences = input_creation(text,tokenizer)
    padded_input_sequences,max_len = padding(input_sequences)
    x,y = create_labels(padded_input_sequences,tokenizer)
    save_tokenizer(tokenizer)
    return x,y,tokenizer,max_len

# --- Run test ---
if __name__ == "__main__":
    x, y, tokenizer, max_len = preprocess()
    print("✅ Preprocessing complete!")
    print("Vocabulary size:", len(tokenizer.word_index))
    print("Max sequence length:", max_len)
    print("X shape:", x.shape)
    print("y shape:", y.shape)
