# app.py

import streamlit as st
import numpy as np
import json
import os
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Predictive Text Generator",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Load Model and Tokenizer ---
# Use Streamlit's caching to load the model and tokenizer only once.
@st.cache_resource
def load_ai_model_and_tokenizer():
    """Loads the pre-trained Keras model and tokenizer."""
    try:
        model_path = os.path.join("model", "text_predictor.h5")
        model = load_model(model_path)

        tokenizer_path = os.path.join("model", "tokenizer.json")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.info("Please make sure the 'model/text_predictor.h5' and 'model/tokenizer.json' files are in the correct directory.")
        return None, None

model, tokenizer = load_ai_model_and_tokenizer()

# --- App UI ---
st.title("✍️ AI Predictive Text Generator")
st.markdown("""
This application uses a trained LSTM neural network to predict and generate text word by word. 
Enter a starting phrase and see what the AI comes up with!
""")

if model is None or tokenizer is None:
    st.stop()

# --- User Inputs ---
seed_text = st.text_input("Enter your starting text:", "")
num_words_to_generate = st.slider("Number of words to generate:", min_value=1, max_value=50, value=10)

if st.button("Generate Text"):
    if not seed_text:
        st.warning("Please enter some starting text.")
    else:
        st.info("Generating text...")
        
        # Get max sequence length from the model's input shape
        max_len = model.input_shape[1] + 1
        generated_text = seed_text
        
        # Use an empty placeholder for dynamic output
        text_placeholder = st.empty()
        text_placeholder.markdown(f"**Generated Text:**\n\n> {generated_text}")

        try:
            for i in range(num_words_to_generate):
                # Tokenize and pad the input text
                token_text = tokenizer.texts_to_sequences([generated_text])[0]
                padded_token_text = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
                
                # Predict the next word
                predicted_probs = model.predict(padded_token_text, verbose=0)
                predicted_index = np.argmax(predicted_probs)
                
                # Map the index back to a word
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted_index:
                        output_word = word
                        break
                
                if output_word:
                    generated_text += " " + output_word
                    # Update the placeholder with the new text
                    text_placeholder.markdown(f"**Generated Text:**\n\n> {generated_text}")
                    time.sleep(0.5) # Add a small delay for a typewriter effect
                else:
                    # If no word is found, stop generating
                    break
            
            st.success("Text generation complete!")

        except Exception as e:
            st.error(f"An error occurred during text generation: {e}")