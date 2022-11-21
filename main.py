#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import tensorflow as tf


# In[8]:


# tf.keras.utils.load_img
# tf.keras.utils.img_to_array


# In[10]:


model=load_model('FruitModel.h5')
labels={0:'apple',1:'banana',2:'beetroot',3:'bell pepper',4:'cabbage',5:'capsicum',6:'carrot', 
        7:'cauliflower',8:'chilli pepper',9:'corn',10:'cucumber',11:'eggplant',12:'garlic',13:'ginger',14:'grapes',
        15:'jalepeno',16:'kiwi',17:'lemon',18:'lettuce',19:'mango',20:'onion',21:'orange',22:'paprika',23:'pear',
        24:'peas',25:'pineapple',26:'pomegranate',27:'potato',28:'raddish',29:'soy beans',30:'spinach',31:'sweetcorn',
        32:'sweetpotato',33:'tomato',34:'turnip',35:'watermelon'}
fruits=['banana','apple','pear','grapes','orange','kiwi','watermelon','pomegranate','mango','tomato']
vegitables=['cucumber','carrot','capsicum','onion','potato','lemon','reddish','beetroot','lettuce','spnich','soy bean',
            'cauliflower','bell pepper','chili pepper','turnip','corn','sweetcorn','sweet potato','peprika','ginger','garlic','peas','eggplant']


# In[112]:


# url = 'https://www.google.com/search?&q=calories in banana'
# req = requests.get(url)
# scrap = BeautifulSoup(req.text, 'html.parser')


# In[1]:


# print(scrap.prettify())
# #"BNeawe iBp4i AP7Wnd"


# In[83]:


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Sorry ! Calories not found")
        print(e)


# In[ ]:


def processes_img(location):
    #from tensorflow.keras.preprocessing.image import load_img,img_to_array
    img=tf.keras.utils.load_img(location,target_size=(224,224,3))
    img=tf.keras.utils.img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()


# In[120]:


def run():
    st.title('Fruits and vegitables Classification')
    img_file=st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
    if img_file is not None:
        img=Image.open(img_file).resize((250,250))
        st.image(img)
        save_image_path=f'.upload_images/{img_file.name}'
        with open(save_image_path,'wb') as f:
            f.write(img_file.getbuffer())
        
        if img_file is not None:
            result=processes_img(save_image_path)
            if result in vegitables:
                st.info('**Category : Vegitable**')
            else:
                st.info('**Category : fruit**')
            st.success('*Predicted : '+result+'**')
            cat=fetch_calories(result)
            if cat:
                st.warning('**'+cat+'**')
run()








