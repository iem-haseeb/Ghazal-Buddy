import string
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import gradio as gr
print("imports done")
# Load the CSV file containing the poetry
file_path = './Roman-Urdu-Poetry.csv'
df = pd.read_csv(file_path)
print("file loaded")
# Assuming the "Poetry" column contains the poetry text
poems = df['Poetry'].dropna().tolist()
poems = poems[0:501]
# Preprocess text to remove punctuation
def preprocess_text(text):
    return text.translate(str.maketrans('', '', string.punctuation))

poems = [preprocess_text(poem) for poem in poems if poem.strip()]
print("poems preprocessed")
# Tokenizer setup
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poems)
total_words = len(tokenizer.word_index) + 1

max_sequence_len = max(len(seq) for seq in tokenizer.texts_to_sequences(poems)) + 1

# Load your trained model
model_path = "./my_trained_model_roman.h5"
model = tf.keras.models.load_model(model_path)
print("model loaded")
# Function to generate the poem
def generate_poem(initial_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([initial_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 2 , padding='pre')
        
        # Predict the next word
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probs)
        
        # Map the predicted index to a word
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        
        # Stop if the model generates an unknown word
        if not predicted_word:
            break
        initial_text += ' ' + predicted_word
    return initial_text


print("predicting")
# Gradio interface
interface = gr.Interface(
    fn=generate_poem,
    inputs=[
        gr.Textbox(label="Initial Text"),
        gr.Slider(label="Next Words to Generate", minimum=10, maximum=500, step=10)
    ],
    outputs=gr.Textbox(label="Generated Poem"),
    title="Ghazal Buddy"
)

interface.launch(share=True)
