import streamlit as st
# import tensorflow as tf
import numpy as np
import pickle

# Load the trained model and mappings
# @st.cache(allow_output_mutation=True)
def load_model_and_mappings():
#     model = tf.keras.models.load_model("roman_urdu_poetry_model.h5")
#     with open("char_to_idx.pkl", "rb") as f:
#         char_to_idx = pickle.load(f)
#     with open("idx_to_char.pkl", "rb") as f:
#         idx_to_char = pickle.load(f)
#     return model, char_to_idx, idx_to_char
    return None, None, None

model, char_to_idx, idx_to_char = load_model_and_mappings()

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, char_to_idx, idx_to_char, num_chars=200, temperature=1.0):
    # maxlen = len(seed_text)
    # generated = seed_text
    # sentence = seed_text
    # for i in range(num_chars):
    #     x_pred = np.zeros((1, maxlen, len(char_to_idx)))
    #     for t, char in enumerate(sentence):
    #         if char in char_to_idx:
    #             x_pred[0, t, char_to_idx[char]] = 1.
    #     preds = model.predict(x_pred, verbose=0)[0]
    #     next_index = sample(preds, temperature)
    #     next_char = idx_to_char[next_index]

    #     generated += next_char
    #     sentence = sentence[1:] + next_char
    # return generated
    return "Ishq ki shama jalti rahi,"

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
            with st.spinner("✨ Creating poetry magic..."):
                # Now num_chars and temperature are defined before this function call
                generated_poetry = generate_text(
                    model=model,
                    seed_text=seed_text,
                    char_to_idx=char_to_idx,
                    idx_to_char=idx_to_char,
                    num_chars=num_chars,
                    temperature=temperature
                )
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
