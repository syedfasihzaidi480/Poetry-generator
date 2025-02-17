import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_data  # remove allow_output_mutation=True
def load_model_and_mappings():
    try:
        model = tf.keras.models.load_model("roman_urdu_poetry_model.h5")
        with open("char_to_idx.pkl", "rb") as f:
            char_to_idx = pickle.load(f)
        with open("idx_to_char.pkl", "rb") as f:
            idx_to_char = pickle.load(f)
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, char_to_idx, idx_to_char, tokenizer
    except Exception as e:
        st.error(f"Error loading model/mappings: {e}")
        st.stop()

model, char_to_idx, idx_to_char, tokenizer = load_model_and_mappings()
max_sequence_length = 50

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_text(model, seed_text, char_to_idx, idx_to_char, num_chars=200, temperature=1.0):
    maxlen = len(seed_text)
    generated = seed_text
    sentence = seed_text
    for _ in range(num_chars):
        x_pred = np.zeros((1, maxlen, len(char_to_idx)))
        for t, char in enumerate(sentence):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_char = idx_to_char[sample(preds, temperature)]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

def generate_poetry(model, seed_text, next_words=150, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=next_words-1, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        probabilities = exp_preds / np.sum(exp_preds)
        predicted_index = np.random.choice(len(probabilities), p=probabilities)
        predicted_word = tokenizer.index_word.get(predicted_index, '')
        seed_text += " " + predicted_word
    return seed_text

def format_poetry(text, num_words_per_line=7):
    words = text.split()
    lines = []
    for i in range(0, len(words), num_words_per_line):
        lines.append(" ".join(words[i:i+num_words_per_line]))
    return "\n".join(lines)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        border-radius: 10px;
    }
    .stTitle {
        color: #1E3D59;
        font-family: 'Arial', sans-serif;
        font-size: 42px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 5px;
        border: 2px solid #1E3D59;
        padding: 10px;
    }
    .stButton > button {
        background-color: #1E3D59;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        width: 100%;
        font-size: 18px;
    }
    .stButton > button:hover {
        background-color: #2E5D79;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI with enhanced layout
st.title("✨ Roman Urdu Poetry Generator")

# Add a decorative separator
st.markdown("<hr style='border: 2px solid #1E3D59; margin: 2rem 0;'>", unsafe_allow_html=True)

# Introduction text with better formatting
st.markdown("""
    <div style='background-color: #f0f5f9; padding: 20px; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: #1E3D59; margin-bottom: 1rem;'>Welcome to the Poetry Generator!</h3>
        <p style='color: #333; font-size: 16px;'>Enter a seed text in Roman Urdu and watch as AI transforms it into a beautiful piece of poetry.</p>
    </div>
""", unsafe_allow_html=True)

# Input section and controls
col1, col2 = st.columns([2, 1])
with col1:
    seed_text = st.text_input("✍️ Enter Your Seed Text", 
                             value="Ishq ki shama jalti rahi,",
                             help="This text will be used as the starting point for the poetry generation")

# Controls in a more organized layout
st.markdown("<br>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    num_chars = st.slider("📝 Length of Poetry", 
                         min_value=50, 
                         max_value=1000, 
                         value=200,
                         help="Adjust the length of the generated poetry")

with col4:
    temperature = st.slider("🎭 Creativity Level", 
                          min_value=0.1, 
                          max_value=1.5, 
                          value=1.0, 
                          step=0.1,
                          help="Higher values make the output more creative but less predictable")

# Generate button moved after sliders
with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    if st.button("🎨 Generate Poetry", help="Click to generate your poetry"):
        if seed_text:
            try:
                with st.spinner("✨ Creating poetry magic..."):
                    # Use generate_poetry for word-level model or generate_text for char-level model
                    if hasattr(model, 'predict_on_batch'):  # Check if it's a word-level model
                        generated_poetry = generate_poetry(
                            model=model,
                            seed_text=seed_text,
                            next_words=num_chars,
                            temperature=temperature
                        )
                    else:  # Use character-level generation
                        generated_poetry = generate_text(
                            model=model,
                            seed_text=seed_text,
                            char_to_idx=char_to_idx,
                            idx_to_char=idx_to_char,
                            num_chars=num_chars,
                            temperature=temperature
                        )
                    generated_poetry = format_poetry(generated_poetry)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("🎭 Please enter some seed text to begin.")

# Display the generated poetry in a styled container
if 'generated_poetry' in locals():
    st.markdown("""
        <div style='background-color: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border: 2px solid #1E3D59;
                    margin-top: 2rem;'>
        <h4 style='color: #1E3D59; margin-bottom: 1rem;'>🎵 Your Generated Poetry</h4>
    """, unsafe_allow_html=True)
    st.text_area("", value=generated_poetry, height=300)
    st.markdown("</div>", unsafe_allow_html=True)
