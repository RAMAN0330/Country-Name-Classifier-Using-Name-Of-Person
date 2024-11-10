# Country-Name-Classifier-Using-Name-Of-Person

This project implements a Recurrent Neural Network (RNN) to classify names based on their language of origin. The model is trained on a dataset of names and is capable of predicting whether a given name belongs to a specific language or region.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

Follow the steps below to set up the project on your local machine.

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/name-language-classification.git
   cd name-language-classification
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv env
   ```
3. Activate the virtual environment:
   - On macOS/Linux:
   ```bash
   source env/bin/activate
   ```
   - on Windows:
   ```bash
   .\env\Scripts\activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
### Usage 
- Predicting the Language of a Name
- After setting up the environment and training the model, you can use the following Streamlit app for interactive classification, run:
  ```bash
  streamlit run app.py
  ```
### Steps to Contribute:

	1.	Fork the repository.
	2.	Create a new branch (git checkout -b feature-branch).
	3.	Commit your changes (git commit -am 'Add new feature').
	4.	Push to the branch (git push origin feature-branch).
	5.	Create a new Pull Request.
