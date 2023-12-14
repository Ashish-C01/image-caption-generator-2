import  tensorflow as tf
from tensorflow.keras.utils import pad_sequences
import pickle
import streamlit as st
import numpy  as np
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input
import logging
tf.get_logger().setLevel(logging.ERROR)

model=tf.keras.models.load_model('model\image_captioning_model.h5')
feature_extractor=tf.keras.models.load_model(r"model\feature_extractor.h5")
with open(r'model\tokenizer_data.pkl','rb') as f:
    pickle_data=pickle.load(f)
    tokenizer=pickle_data['tokenizer']
    idx_to_word=pickle_data['word_mapping']
max_length=35

def generate_caption(img):
    in_text = 'startseq'
    new_img=Image.open(img)
    image=np.asarray(new_img.resize((299,299)))
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image=preprocess_input(image)
    feature=feature_extractor.predict(image,verbose=0)
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        if yhat not in idx_to_word.keys():
            break
        else:
            word=idx_to_word[yhat]
            if word == 'endseq':
                break
            in_text += " " + word
            
      
    return in_text.replace("startseq",'')

st.title("Image Caption Generator")
img=st.file_uploader("Upload image",type=["png","jpg","jpeg"],)
if img is not None:
    st.image(Image.open(img),width=300)
    
if st.button("Generate Caption"):
    if img is not None:
        st.write(generate_caption(img))
    else:
        st.write("Please upload an image")