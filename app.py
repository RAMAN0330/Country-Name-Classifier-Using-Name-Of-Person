# streamlit_app.py

import streamlit as st
import torch
from model import RNN  # Import your model class and helper functions
import string
import re

def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

# Load the model
model_path = "classify_names.pth"

# Initialize model (make sure to use the same parameters as in training)
n_hidden = 128  # Example value; match this with the one you used
n_categories = 18
n_letters = 57  # Set according to the number of name origins in your dataset
rnn = RNN(n_letters, n_hidden, n_categories)
rnn.load_state_dict(torch.load(model_path,weights_only=True))
rnn.eval()

all_categories = ['Czech',
 'German',
 'Arabic',
 'Japanese',
 'Chinese',
 'Vietnamese',
 'Russian',
 'French',
 'Irish',
 'English',
 'Spanish',
 'Greek',
 'Italian',
 'Portuguese',
 'Scottish',
 'Dutch',
 'Korean',
 'Polish']

# Function to evaluate name
def evaluate_name(name):
    # Convert name to tensor format compatible with the model
    def line_to_tensor(line):
        tensor = torch.zeros(len(line), 1, len("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'"))
        
        for li, letter in enumerate(line):
            # Find the index of the letter in the alphabet
            index = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'".find(letter)
            
            # If letter is not found, you can either skip or set a default behavior (like setting index to a default value)
            if index == -1:
                # Handle the case where the letter is not found, you could skip it or log an error
                print(f"Warning: Character '{letter}' not found in alphabet sequence.")
                continue
            
            # Set the corresponding index in the tensor
            tensor[li][0][index] = 1
        
        return tensor

    name_tensor = line_to_tensor(name)
    output = evaluate(name_tensor)
    category, _ = category_from_output(output)
    return category

# Function to clean and process name input
def clean_name_input(name):
    name = name.strip()
    name = re.sub(f'[^{string.ascii_letters}]', '', name)
    return name.capitalize()

# Streamlit UI
st.set_page_config(page_title="Name Origin Classifier", page_icon="🌍", layout="centered")

st.title("🌍 Name Origin Classifier")
st.write("Enter a name to find out its origin. This model predicts the language or region associated with the name (e.g., Spanish, Japanese, etc.).")

# Input form for name
name = st.text_input("Enter a name:", max_chars=30)

if st.button("Classify Name"):
    if name:
        name = clean_name_input(name)
        predicted_origin = evaluate_name(name)
        
        st.success(f"The origin of **{name}** is predicted to be: **{predicted_origin}**.")
    else:
        st.warning("Please enter a valid name.")

st.write("### About this App")
st.write("This app uses a Recurrent Neural Network (RNN) model to predict the origin of a given name. The model was trained on a dataset with names from various origins, allowing it to classify based on learned patterns.")

st.write("Built with ❤️ using Streamlit and PyTorch")
