# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 21:35:24 2022

@author: TAR
"""
# import libs
import streamlit as st
import easyocr
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import numpy as np


# Variable easyOcr class
classLang = ['en', 'fr']
classGpu = [True, False]
classModel_storage_directory = []
classDownload_enabled = [True, False]
classUser_network_directory = []
classRecog_network = []
classDetector = [True, False]
classRecognizer = [True, False]


st.title("Text Recognition with EasyOCR")


with st.sidebar:
    
    Lang = st.selectbox('Langue', classLang)
    
    batch_size = st.slider("batch size", min_value=1, max_value=10)
    min_size = st.slider("min pixel size", min_value=1, max_value=100, value=10)
    decoder = st.selectbox("decoder",['greedy','beamsearch','wordbeamsearch'])
    beamWidth  = st.slider("beamWidth", min_value=1, max_value=10, value=5)
    

# load picture
picture = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if picture is not None:
    
    @st.cache(suppress_st_warning=True)
    def EasyOCR(picture, Lang, batch_size, min_size, decoder, beamWidth):
        # instanciation function easyOcr
        reader = easyocr.Reader([Lang])
        # Doing OCR. Get bounding boxes.
        bounds = reader.readtext(picture)
    
        return(bounds)
    
    @st.cache    
    # Draw bounding boxes
    def draw_boxes(image, bounds, color='red', width=4):
        draw = ImageDraw.Draw(image)
        for bound in bounds:
            p0, p1, p2, p3 = bound[0]
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        return(image)
        
    @st.cache
    def SearchWord(bounds, word):
        # création df with datass
        df = pd.DataFrame(bounds, columns=["bounding", "text", "prob"])
        # lower char
        df = df.apply(lambda x: x.astype(str).str.lower())
        # searching word
        w = df['text'].str.match(word)
        # index value
        wordIndex = w[w].index
        # check one line foud
        if wordIndex.shape[0] == 1:
            line = df.iloc[wordIndex.values]
        
        # search closet value
        wordLocalisation = line['bounding']
        # remove char
        wordLocalisation = line['bounding'].replace({'\[|\]' : ''}, regex=True)
        # split coordinates
        wordLocalisation = pd.DataFrame(wordLocalisation.str.split(',', expand=True)) 
        # type var modification
        wordLocalisation = wordLocalisation.astype('float')
        # var name modified
        wordLocalisation = wordLocalisation.rename(columns={0:'x', 1:'y', 2:'w', 5:'h'})
        
        # remove char, split, type and mane modified
        allLocalisations = df['bounding'].replace({'\[|\]' : ''}, regex=True)
        allLocalisations = pd.DataFrame(allLocalisations.str.split(',', expand=True))
        allLocalisations = allLocalisations.astype('float')
        allLocalisations = allLocalisations.rename(columns={0:'x', 1:'y', 2:'w', 5:'h'})
        
        # calc euclid distance and sort value
        allLocalisations['euclid'] = np.sqrt((allLocalisations.x - wordLocalisation.x.values).pow(2) + (allLocalisations.y - wordLocalisation.y.values).pow(2))
        allLocalisations = allLocalisations[['x', 'y', 'w', 'h', 'euclid']].sort_values(by='euclid', ascending=True)
    
        return(df.text.iloc[allLocalisations[0:1].index.values].values,
               df.text.iloc[allLocalisations[1:2].index.values].values)
        
    @st.cache
    def picToArray(pic):
        im = Image.open(pic)
        arr = np.array(im)
        return(arr)
        
    
    # function application
    im = Image.open(picture)
    array = picToArray(picture)
    bounds = EasyOCR(array, Lang, batch_size, min_size, decoder, beamWidth)

    pic = draw_boxes(im, bounds)
    
    # showing pictures
    st.image(pic, caption="The caption")
    
    # showing data
    df = pd.DataFrame(bounds, columns=["bounding", "text", "prob"])
    st.dataframe(df)