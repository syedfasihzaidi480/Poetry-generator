import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load the trained model and mappings
@st.cache(allow_output_mutation=True)
def load_model_and_mappings():
    model = tf.keras.models.load_model("roman_urdu_poetry_model.h5")
    with open("char_to_idx.pkl", "rb") as f:
        char_to_idx = pickle.load(f)
    with open("idx_to_char.pkl", "rb") as f:
        idx_to_char = pickle.load(f)
    return model, char_to_idx, idx_to_char

model, char_to_idx, idx_to_char = load_model_and_mappings()

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, char_to_idx, idx_to_char, num_chars=200, temperature=1.0):
    maxlen = len(seed_text)
    generated = seed_text
    sentence = seed_text
    for i in range(num_chars):
        x_pred = np.zeros((1, maxlen, len(char_to_idx)))
        for t, char in enumerate(sentence):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

# Streamlit UI
st.title("Roman Urdu Poetry Generator")
st.write("Enter a seed text in Roman Urdu and let the model generate a full piece of poetry.")

seed_text = st.text_input("Seed Text", value="Ishq ki shama jalti rahi,")
num_chars = st.slider("Number of characters to generate", min_value=50, max_value=1000, value=200)
temperature = st.slider("Temperature (creativity control)", min_value=0.1, max_value=1.5, value=1.0, step=0.1)

if st.button("Generate Poetry"):
    if seed_text:
        with st.spinner("Generating..."):
            generated_poetry = generate_text(model, seed_text, char_to_idx, idx_to_char, num_chars, temperature)
        st.text_area("Generated Roman Urdu Poetry", value=generated_poetry, height=300)
    else:
        st.error("Please enter some seed text.")
