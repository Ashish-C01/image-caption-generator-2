# Image Caption Generator [Application](https://huggingface.co/spaces/ashish-001/Image_Caption_Generator)
This model will generate a caption for the input image. This model was trained on Flickr30k dataset.
[Link](https://huggingface.co/spaces/ashish-001/Image_Caption_Generator)

This project involves designing and implementing an image caption generator model that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model leverages the state-of-the-art Xception model in CNN to extract high-level features from images, enabling the automatic generation of descriptive captions based on the input image.

## Key Features

- **Convolutional Neural Networks (CNN)**: Utilizes the Xception model to extract high-level image features.
- **Long Short-Term Memory (LSTM) Networks**: Generates descriptive captions based on the features extracted from images.
- **Automatic Caption Generation**: Produces human-like captions for a wide variety of images.

## Model Architecture

1. **Feature Extraction**: The Xception model, a powerful CNN architecture, is used to extract detailed features from the input images.
2. **Sequence Generation**: An LSTM network is employed to process the extracted features and generate descriptive captions.

## Installation

## To set up the project locally, follow these steps on Windows
1. Create a virtual environment 
```
python -m venv "environment name"
```
2. Activate the virtual environment
```
"environment name"\Scripts\activate
```
3. Install all required libraries
```
pip install -r requirements.txt
```
4. Run the program
```
streamlit run frontend\main.py
```

## Images
![Alt text](<image.png>)
