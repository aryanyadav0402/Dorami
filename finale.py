#!/usr/bin/env python
# coding: utf-8

# ## NER

# In[ ]:





# In[50]:


#NER REQ.
import spacy
import pickle
import re
import numpy as np
from spacy.tokens import Span
from sentence_transformers import SentenceTransformer
model_price_senti = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#INTENT REQ.
#import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.backend as K
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras import Model
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers import Bidirectional, LSTM
import pickle  
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import sklearn
import contextlib
from contextlib import contextmanager
import sys, os

#QnA
from sentence_transformers import SentenceTransformer
model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Action Code
import random


# In[51]:


pat_label = {"price_senti":"\w+[\s]\w+[\s][Rr][\s]?[Ss][\.]?[\s]?\d*[,]?\d*",
             "price_range":"[Rr][\s]?[Ss][\.]?[\s]?\d*[,]?\d*[\s]?[t|a]?[o|n]?[-|d]?[\s]?[Rr][\s]?[Ss][\.]?[\s]?\d*[,]?\d*",
             "weight_senti":"\w+[\s]\w+[\s]\w+\d*[\.]?\d*[\s][Kk][Gg|Ii][Ll]?[Ll]?[Oo]?[Gg]?[Rr]?[Aa]?[Mm]?[Ss]?",
             "weight_range":"\d*[\.]?\d*[\s][Kk][Gg|Ii][Ll]?[Ll]?[Oo]?[Gg]?[Rr]?[Aa]?[Mm]?[Ss]?[\s]?[t|a]?[o|n]?[-|d]?[\s]?\d*[\.]?\d*[\s][Kk][Gg|Ii][Ll]?[Ll]?[Oo]?[Gg]?[Rr]?[Aa]?[Mm]?[Ss]?",
             "size_senti":"\w+[\s]\w+[\s]\w+\d*[\.]?\d*[\s][I|i][N|n][C|c][H|h][e|E]?[s|S]?",
             "size_range":"\d*[\.]?\d*[\s][I|i][N|n][C|c][H|h][e|E]?[s|S]?[\s]?[t|a]?[o|n]?[-|d]?[\s]?\d*[\.]?\d*[\s][I|i][N|n][C|c][H|h][e|E]?[s|S]?",
             "ram_senti":"\w+[\s]\w+[\s]\w+\d*[\.]?\d*[\s][Gg][Bb|Ii][Gg]?[Ss|Aa]?[Bb]?[Ii|Yy]?[Tt]?[Ee]?[Ss]?",
             "ram_range":"\d*[Gg][Bb|Ii][Gg]?[Ss|Aa]?[Bb]?[Ii|Yy]?[Tt]?[Ee]?[Ss]?[\s]?[t|a]?[o|n]?[-|d]?[\s]?\d*[\s][Gg][Bb|Ii][Gg]?[Ss|Aa]?[Bb]?[Ii|Yy]?[Tt]?[Ee]?[Ss]?"
             }

def classify_query_category(query,ent_label):
    regex_nlp = pickle.load(open("regex_nlp.pkl","rb"))
    doc_reg = regex_nlp(query)
    
    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    number_ent_query = []
    query_ents = []

    for ent in doc_reg.ents:
      if ent.label_ == ent_label:
        number_ent_query.append(ent.text)
    
    if(len(number_ent_query)==1):
      pattern_query = {"label":str(str(ent_label)+"_senti"), "pattern": pat_label[str(ent_label+"_senti")]}
      label = str(ent_label+"_senti")
    elif(len(number_ent_query)!=1) :
      pattern_query = {"label":str(str(ent_label)+"_range"), "pattern": pat_label[str(ent_label+"_range")]}
      label = str(ent_label+"_range")
        
    original_ents = list(doc_reg.ents)
    
    mwt_ents = []
    
    for match in re.finditer(pattern_query["pattern"], doc_reg.text):
        start, end = match.span()
        span = doc_reg.char_span(start, end)
        if span is not None:
            mwt_ents.append((span.start, span.end, span.text))
            
#     s = match.start()
#     e = match.end()
    
#     print('String match: "%s" at %d:%d' % (text[s:e], s, e))
    
    for ent in mwt_ents:
        start, end, name = ent
        per_ent = Span(doc_reg, start, end, label)
        original_ents.append(per_ent)
        
#     print(mwt_ents)
    ent_text_4_comparsion = str(mwt_ents[0][2])
        
#     for ent in original_ents:
#          print ("Entity text:",ent.text,"|| Entity label:", ent.label_)
    
    text_query_above = str("higher than")
    text_query_below = str("lower than")
    text_query_around = str("nearly around")
    sentences = [text_query_above,text_query_below,text_query_around]
    sim_values = []
    embedding = model_price_senti.encode(sentences)
    queryyy = str(ent_text_4_comparsion)
    query_vec = model_price_senti.encode([queryyy])[0]
    for sent in sentences:
        sim = cosine(query_vec, model_price_senti.encode([sent])[0])
        sim_values.append(sim)
    if(len(number_ent_query)==1):
        if max(sim_values) == sim_values[0]:
            sim_class = str(ent_label+"_above")
        elif max(sim_values) == sim_values[1]:
            sim_class = str(ent_label+"_below")
        elif max(sim_values) == sim_values[2]:
            sim_class = str(ent_label+"_around")
    else:
        sim_class = str(ent_label+"_range")
    return sim_class


# In[52]:


def list_query_ents(query):
    regex_nlp = pickle.load(open("regex_nlp.pkl","rb"))
    ents_list = ["price","size","weight","ram"]
    doc_reg = regex_nlp(str(query))
    query_ents = []
    for ent in doc_reg.ents:
            query_ents.append((ent.text, ent.start_char, ent.end_char, ent.label_))
#     for ent in doc_reg.ents:
#         if ent.label_ == ent_label:
#             num_ent_queries.append(ent.text)
    for entity in ents_list:
        num_ent_queries = []
        for ent in doc_reg.ents:
            if ent.label_ == entity:
                num_ent_queries.append((ent.text, ent.start_char, ent.end_char, ent.label_))
        if(len(num_ent_queries)!=0):
            query_ents.append((classify_query_category(query,entity), 0, 0, str(entity+"_category")))
            
#         print("Entity_text - ",ent.text,"=> Entity_class - ", ent.label_)
#     print(non_price_ents)
    return query_ents


# In[53]:


def extract_entities(user_input, visualize = False):
    # Loading it in
    laptop_nlp = pickle.load(open("laptop_big_nlp.pkl", "rb"))
   
    doc = laptop_nlp(user_input)
    extracted_entities = []
    ents = list_query_ents(user_input)

    # These are the objects you can take out
    for ent in doc.ents:
        if ent.text == '.':
            pass
        else:
            extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))
    for i in ents:
        extracted_entities.append(i)
        
    # If you want to visualize
    # if visualize == True:
    #     # Visualizing with displaCy how the document had it's entity tagged (runs a server)
    #     colors = {"product_type": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    #     options = {"ents": ["product_type"], "colors": colors}
    #     html = displacy.render(doc, style = 'ent', options = options)
    #     display(HTML(html));
    #     displacy.serve(doc, style="ent", options=options)
    #     displacy.serve(doc, style="ent")
    return extracted_entities


# #INTENT

# In[54]:


module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.load(module_url)


# In[55]:


input_text1 = Input(shape=(512,))
x = Dense(256, activation='relu')(input_text1)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = Dropout(0.4)(x)
dense_layer = Dense(128, name='dense_layer')(x)
norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

model=Model(inputs=[input_text1], outputs=norm_layer)



# In[56]:


# Input for anchor, positive and negative images
in_a = Input(shape=(512,))
in_p = Input(shape=(512,))
in_n = Input(shape=(512,))

# Output for anchor, positive and negative embedding vectors
# The nn4_small model instance is shared (Siamese network)
emb_a = model(in_a)
emb_p = model(in_p)
emb_n = model(in_n)

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a-p), axis=-1)
        n_dist = K.sum(K.square(a-n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

# Layer that computes the triplet loss from anchor, positive and negative embedding vectors
triplet_loss_layer = TripletLossLayer(alpha=0.4, name='triplet_loss_layer')([emb_a, emb_p, emb_n])

# Model that can be trained with anchor, positive negative images
nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)


# In[57]:


def get_triplets(unique_train_label,map_train_label_indices):
      label_l, label_r = np.random.choice(unique_train_label, 2, replace=False)
      a, p = np.random.choice(map_train_label_indices[label_l],2, replace=False)
      n = np.random.choice(map_train_label_indices[label_r])
      return a, p, n

def get_triplets_batch(k,train_set,unique_train_label,map_train_label_indices,embed):

    while True:
      idxs_a, idxs_p, idxs_n = [], [], []
      for _ in range(k):
          a, p, n = get_triplets(unique_train_label,map_train_label_indices)
          idxs_a.append(a)
          idxs_p.append(p)
          idxs_n.append(n)

      a=train_set.iloc[idxs_a].values.tolist()
      b=train_set.iloc[idxs_p].values.tolist()
      c=train_set.iloc[idxs_n].values.tolist()

      a = embed(a)
      p = embed(b)
      n = embed(c)
        # return train_set[idxs_a], train_set[idxs_p], train_set[idxs_n]
      yield [a,p,n], []


# In[58]:


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)


# In[59]:

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
with suppress_stdout():
    nn4_small2_train.load_weights(latest)


# In[60]:


# save the model to disk
filename = 'svc.sav'
# load the model from disk
svc = pickle.load(open(filename, 'rb'))
#result = svc.score(X_test, Y_test)


# ## QNA

# In[61]:


df=pd.read_csv('dataset 1 - Sheet1.csv')


# In[62]:


#Data we want to encode.
sentences = df["Question"]

#Encoding the Data
##embedding = model1.encode(sentences)


# In[63]:


with open('embedding.pickle', 'rb') as pkl:
    embedding = pickle.load(pkl)


# In[64]:


#Defining cosine similarity function
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# In[65]:


#Using similarity values to answer queries

def Q_A(query):
    # while(True):
      #query = input("User: ")
    query_vec = model1.encode([query])[0]
    sim = []
    for sent in sentences:
        sim.append(cosine(query_vec, model1.encode([sent])[0]))
    if(max(sim)>0.4):
        return df['Answer'][sim.index(max(sim))]
    else:
        return 'Sorry, I could not understand.'
#     print(res)
      #     continue
      # if(df['Answer'][sim.index(max(sim))]=="Thank you! See you again."):
      #     break


# In[66]:


# res=dict()


# ## Action

# In[67]:


dflist = pd.read_csv('prod_gaming_laptops - flipkart-scraper_8077.csv')
dflist.title = dflist.title.map(lambda x: x.lower())
dflist.insert(0, 'product', dflist.title.map(lambda x: x.split('-')[0]))
dflist.insert(0, 'brand', dflist.title.map(lambda x: x.split()[0].lower()))


# In[68]:


def intents_ent(uq):
    user_query1 = model.predict(np.array(embed(np.array(pd.Series(uq).values.tolist()))['outputs']))
    intent = svc.predict(user_query1)
    entities = [(entity[0].lower(), entity[3].lower()) for entity in extract_entities(uq)]
    tups = entities
    dictionary = {}
    entity_dict = Convert(tups, dictionary)

    return intent, entity_dict


# In[69]:


def Convert(tup, di):
    for b, a in tup:
        di.setdefault(a, []).append(b)
    return di


# In[70]:


# for finding the specific product(s) from the database
def sql_search(entity_dict):
    # df2 = df[df['title'].str.contains(entities[])
    value1 = list(entity_dict.values())[0]
    Bool = dflist['title'].str.contains(value1[0])

#     print(entity_dict)
    for key, value in entity_dict.items():
      if key == 'ram':
        Bool1 = dflist['title'].str.contains(value[0] + "/") 
#         Bool1 = dflist['title'].str.contains(value[0] + "/") | dflist['specs'].str.contains(value[0])
      else:
        Bool1 = dflist['title'].str.contains(value[0])
#         Bool1 = dflist['title'].str.contains(value[0]) | dflist['specs'].str.contains(value[0])
      Bool1 = (Bool & Bool1)
      Bool = Bool1
    
    df2 = dflist[Bool]
    return df2
    # output = 'Dorami: These are all the'
    # for key in entity_dict.keys():
    #   output += ' ' + entity_dict[key][0]
  
    # output += ' in our catalogue. You can choose any one of them or even give me more specifications that you want so I can narrow down the results for you!'
    # print(output, df2.title, sep='\n')


# In[71]:


# to capture missing entities
def missing_entity(missing_entity, product_type, entity_dict):
    
    a = "missing_ent"
    past_entity_dict = entity_dict.copy()
    if '_' in missing_entity:
        
        return 'Is there any specific {} in which you want the {} or anything will do?'.format(' '.join(missing_entity.split('_')), product_type)
    else:

        return 'Is there any specific {} in which you want the {} or anything will do?'.format(missing_entity, product_type)
#      print('Dorami: Is there any specific {} in which you want the {} or anything will do?'.format(missing_entity, product_type))
#       p = 'Dorami: Is there any specific {} in which you want the {} or anything will do?'
#       res.append(p)
#       return res
    #user_query = input('User: ')
    
    # capturing intents and entities
    

#     intent, missing_entity_dict = intents_ent(user_query)
#     print(intent, missing_entity_dict)

#     if intent == 'no':
#       return dict()
#     else:
#       return missing_entity_dict


# In[72]:


def comp_search(entity_dict):
    # df2 = df[df['title'].str.contains(entities[])
    value1 = list(entity_dict.values())[0][0]
    value2 = list(entity_dict.values())[0][1]
    Bool1 = dflist['title'].str.contains(value1)
    Bool2 = dflist['title'].str.contains(value2)
    df1 = dflist[Bool1]
    df2 = dflist[Bool2]
    # print(df1.title)
    # print(df2.title)
    value1 = list(entity_dict.values())[1]
    Bool = df1['title'].str.contains(value1[0])
  
    for value in entity_dict.values(): 
#       Bool1 = df1['title'].str.contains(value[0]) | df1['specs'].str.contains(value[0])
      Bool1 = df1['title'].str.contains(value[0])
      Bool1 = (Bool & Bool1)
      Bool = Bool1
    
    df11 = df1[Bool]
    
    value2 = list(entity_dict.values())[1]
    print(value2)
    Bool = df2['title'].str.contains(value2[1])
    # print(df11)
    # print('2nd')
    for value in entity_dict.values(): 
      # if(value not in value2[0] ):
#         Bool1 = df2['title'].str.contains(value[0]) | df2['specs'].str.contains(value[0])
        Bool1 = df2['title'].str.contains(value[0])
        Bool1 = (Bool | Bool1)
        Bool = Bool1
    # print(Bool)
    df22 = df2[Bool]
    #print(df22)
#     print('Dorami: Product 1 specs', df11.title , df11.specs , df11.price, sep='\n' )
#     print('Dorami: Product 2 specs', df22.title , df22.specs , df22.price , sep='\n' )
    return [('Dorami: Product 1 specs', df11.title , df11.price), ('Dorami: Product 2 specs', df22.title, df22.price)]


# In[73]:


past_intents = []
past_entities = []
past_entity_dict = dict()
a = None
b = None


# In[74]:


# res={}
# res['data']='The price of the requested products are shown below:-'
# res['title']=["a","b"]
# res['price']=[3,4]
# res


# In[93]:


def chatbot(user_query):
    try:
        # initializing the dictionary containing results to be returned to the frontend
        res = dict()
        
        # capturing intents and entities
        intent, entity_dict = intents_ent(user_query)
#         print(intent, entity_dict)
        
        
        # to capture missing product name in comparison info search
        if past_intents != [] and past_intents[-1] == 'Comparison_info':
            past_entity_dict = past_entities[-1]
            past_entity_dict.update(entity_dict)
            res['data'] = comp_search(past_entity_dict)
            return res


#         # for missing_entity function
#         if a == "missing_ent":
#             if intent == "no":
#                 #print('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title, sep='\n')
#                 a = None
#                 pass
#             else:
#                 past_entity_dict.update(entity_dict)
#                 entity_dict = past_entity_dict.copy()

#         # just to check if code capturing right intents and entitites
#        # print(intent,entity_dict)

#         # when Dorami asks the user more details to narrow the search results
#         if past_intents != [] and past_intents[-1] == 'Buy_info':
#           #print(past_intents[-1] )
#           past_entity_dict = past_entities[-1]
#           past_entity_dict.update(entity_dict)
#           if 'brand' not in past_entity_dict.keys() and b != 'brand':
#               res['data'] = missing_entity('brand', past_entity_dict['product_type'][0], past_entity_dict)
#               b = 'brand'
#               return res
#     #           past_entity_dict.update(missing_entity_dict)
#           if 'product_name' not in past_entity_dict.keys():
#               res['data'] = missing_entity('product_name', past_entity_dict['product_type'][0], past_entity_dict)
#     #           past_entity_dict.update(missing_entity_dict)
#           if 'colour' not in past_entity_dict.keys():
#               res['data'] = missing_entity('colour', past_entity_dict['product_type'][0], past_entity_dict)
#     #           past_entity_dict.update(missing_entity_dict)

#           # appending spec_info intent at the end of intents list to avoid going into this if clause again in the next chatbot func iteration
#           past_intents.append('spec_info')
#     #       return(past_entity_dict)
#     #      print(past_entity_dict)
#           query_df = sql_search(past_entity_dict)
#     #       return('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title, sep='\n')
#           p=('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title)
#           res.append(p)
#           return res

        # when Dorami asks the user more details to narrow the search results
        
#         if past_intents != [] and (past_intents[-1] == 'Buy_info' or past_intents[-1] == 'similar_product') and intent not in ['Bye', 'yes', 'Q_A', 'Price_info', 'similar_product', 'Comparison_info']:
#             print(past_intents[-1])
#             past_entity_dict = past_entities[-1]
#             past_entity_dict.update(entity_dict)
#             if 'product_type' not in past_entity_dict.keys():
#                 # for now since we are only dealing in laptops
#                 past_entity_dict['product_type'] = 'laptop'
#             if 'brand' not in past_entity_dict.keys():
#                 res['data'] = missing_entity('brand', past_entity_dict['product_type'][0], past_entity_dict)
#             if 'product_name' not in past_entity_dict.keys():
#                 res['data'] = missing_entity('product_name', past_entity_dict['product_type'][0], past_entity_dict)
#             if 'colour' not in past_entity_dict.keys():
#                 res['data'] = missing_entity('colour', past_entity_dict['product_type'][0], past_entity_dict)

#           # appending _ intent at the end of intents list to avoid going into this if clause again in the next chatbot func iteration
# #             past_intents.append('_')
# #             past_entities.append(past_entity_dict)
# #             print(past_entity_dict)

# #             query_df = sql_search(past_entity_dict)
# #             print('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title, sep='\n')
            

#         if past_intents != [] and (past_intents[-2] == 'Buy_info' or past_intents[-2] == 'similar_product') and 'brand' in list(entity_dict.keys()):
#             past_entity_dict.update(entity_dict)
        
#         if past_intents != [] and (past_intents[-3] == 'Buy_info' or past_intents[-3] == 'similar_product') and 'product_name' in list(entity_dict.keys()):
#             past_entity_dict.update(entity_dict) 
            
#         if past_intents != [] and (past_intents[-4] == 'Buy_info' or past_intents[-4] == 'similar_product') and 'colour' in list(entity_dict.keys()):
#             past_entity_dict.update(entity_dict)
#             res['data'] = sql_search(past_entity_dict)
            
            
            

    #     # when Dorami asks the user more details to narrow the search results
    #     elif len(past_intents) > 1 and (past_intents[-2] == 'Buy_info' or past_intents[-2] == 'similar_product') and intent not in ['Bye', 'yes']:
    #       print(past_intents[-2] )
    #       past_entity_dict = past_entities[-2]
    #       past_entity_dict.update(entity_dict)
    #       if 'product_type' not in past_entity_dict.keys():
    #           past_entity_dict['product_type'] = 'laptop'
    #       if 'brand' not in past_entity_dict.keys():
    #           missing_entity_dict = missing_entity('brand', past_entity_dict['product_type'][0], past_entity_dict)
    #           past_entity_dict.update(missing_entity_dict)
    #       if 'product_name' not in past_entity_dict.keys():
    #           missing_entity_dict = missing_entity('product_name', past_entity_dict['product_type'][0], past_entity_dict)
    #           past_entity_dict.update(missing_entity_dict)
    #       if 'colour' not in past_entity_dict.keys():
    #           missing_entity_dict = missing_entity('colour', past_entity_dict['product_type'][0], past_entity_dict)
    #           past_entity_dict.update(missing_entity_dict)

    #       # appending spec_info intent at the end of intents list to avoid going into this if clause again in the next chatbot func iteration
    #       past_intents.append('_')
    #       past_entities.append(past_entity_dict)
    #       print(past_entity_dict)

    #       query_df = sql_search(past_entity_dict)
    #       print('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title, sep='\n')
    #       chatbot()

        # storing the intents and entities for use in above scenario and dialogue context tracking
        past_intents.append(intent)
        past_entities.append(entity_dict)


        # q_a intent
        if intent == 'Q_A':
            res['data'] = Q_A(user_query)
            return res


        # buy_info
        elif intent == 'Buy_info':
            flag = 0
            if 'laptop' in entity_dict['product_type']:
                # print(entity_dict)
    #             dict_ents_iterable = ['price','size','ram','weight']
    #             for entity in dict_ents_iterable:


                if 'price_category' in entity_dict:
                    flag += 1
    #                 print("t57")
                    if 'price_above' in entity_dict['price_category']:
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['price'][0]))
    #                   print("t57")
    #                   print(price_cap_int) 
    #                   print(entity_dict)
                      entity_dict.pop('price')
                      entity_dict.pop('price_category')
    #                   print(entity_dict)
                      dfprice = sql_search(entity_dict)
    #                   print(dfprice)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[] 
                      # for row in dfprice:
    #                   print(res)
                      for (row,series) in dfprice.iterrows():
    #                         print('yoooooo')    
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[5]))
    #                     print(price_dat_int)
                        if price_dat_int > price_cap_int:
    #                       print("yes+++")
            #               return(series[0],series[3])
    #                       print(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[5])
                        #  print(series[0],series[3+dict_ents_iterable.index(entity)])
                        #chatbot()
                    elif ('price_below') in entity_dict['price_category']:
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['price'][0]))
    #                   print(price_cap_int)
                      entity_dict.pop('price')
                      entity_dict.pop('price_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      # for row in dfprice:
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[5]))
    #                     print(price_dat_int)
                        if price_dat_int < price_cap_int:
            #               return(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[5])
                    elif ('price_around') in entity_dict['price_category']:
                      print("t59")
                      t = 0
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['price'][0]))
                      entity_dict.pop('price')
                      entity_dict.pop('price_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      prod_ar = []
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[5]))
                        if abs(int(price_dat_int) - int(price_cap_int))<abs(int(t)-int(price_cap_int)):
                          t = price_dat_int
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[5]))
                        if t == price_dat_int:
                          res['title'].append(series[2])
                          res['entity'].append(series[5])

                    elif ('price_range') in entity_dict['price_category']:
                      a = ''.join(filter(lambda i: i.isdigit(), entity_dict['price'][0]))
                      b = ''.join(filter(lambda i: i.isdigit(), entity_dict['price'][1]))
                      price_cap_int_low = min(a,b)
                      price_cap_int_high = max(a,b)
                      entity_dict.pop('price')
                      entity_dict.pop('price_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      # for row in dfprice:
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[5]))
                        if price_cap_int_low <price_dat_int < price_cap_int_high:
            #               return(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[5])


                elif ('size_category') in entity_dict:
                    flag += 1
                    print("t58")
                    if ('size_above') in entity_dict['size_category']:
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['size'][0]))
                      print("t58")
                      entity_dict.pop('size')
                      entity_dict.pop('size_category')
    #                       print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[] 
                      # for row in dfprice:
                      for (row,series) in dfprice.iterrows():
    #                         print('yoooooo')    
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[6]))
    #                         print(price_dat_int)
                        if price_dat_int > price_cap_int:
            #               return(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[6])
                        #  print(series[0],series[3+dict_ents_iterable.index(entity)])
                        #chatbot()
                    elif ('size_below') in entity_dict['size_category']:
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['size'][0]))
                      print("t58")
    #                   print(price_cap_int)
                      entity_dict.pop('size')
                      entity_dict.pop('size_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      # for row in dfprice:
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[6]))
    #                     print(price_dat_int)
                        if price_dat_int < price_cap_int:
            #               return(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[6])
                    elif ('size_around') in entity_dict['size_category']:
                      t = 0
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['size'][0]))
                      entity_dict.pop('size')
                      entity_dict.pop('size_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      prod_ar = []
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[6]))
                        if abs(int(price_dat_int) - int(price_cap_int))<abs(int(t)-int(price_cap_int)):
                          t = price_dat_int
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[6]))
                        if t == price_dat_int:
                            res['title'].append(series[2])
                            res['entity'].append(series[6])
    #                         print(prod_ar)
    #                         res['title'].append(series[0])
    #                         res['entity'].append(series[4])

                    elif ('size_range') in entity_dict['size_category']:
                      a = ''.join(filter(lambda i: i.isdigit(), entity_dict['size'][0]))
                      b = ''.join(filter(lambda i: i.isdigit(), entity_dict['size'][1]))
                      price_cap_int_low = min(a,b)
                      price_cap_int_high = max(a,b)
                      entity_dict.pop('size')
                      entity_dict.pop('size_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      # for row in dfprice:
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[6]))
                        if price_cap_int_low <price_dat_int < price_cap_int_high:
            #               return(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[6])



                elif ('ram_category') in entity_dict:
                    flag += 1
                    print("t58")
                    if ('ram_above') in entity_dict['ram_category']:
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['ram'][0]))
                      print("t58")
                      entity_dict.pop('ram')
                      entity_dict.pop('ram_category')
    #                       print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[] 
                      # for row in dfprice:
                      for (row,series) in dfprice.iterrows():
    #                         print('yoooooo')    
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[7]))
    #                         print(price_dat_int)
                        if price_dat_int > price_cap_int:
            #               return(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[7])
                        #  print(series[0],series[3+dict_ents_iterable.index(entity)])
                        #chatbot()
                    elif ('ram_below') in entity_dict['ram_category']:
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['ram'][0]))
                      print("t58")
    #                   print(price_cap_int)
                      entity_dict.pop('ram')
                      entity_dict.pop('ram_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      # for row in dfprice:
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[7]))
    #                     print(price_dat_int)
                        if price_dat_int < price_cap_int:
            #               return(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[7])
                    elif ('ram_around') in entity_dict['ram_category']:
                      t = 0
                      price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict['ram'][0]))
                      entity_dict.pop('ram')
                      entity_dict.pop('ram_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      prod_ar = []
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[7]))
                        if abs(int(price_dat_int) - int(price_cap_int))<abs(int(t)-int(price_cap_int)):
                          t = price_dat_int
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[7]))
                        if t == price_dat_int:
                            prod_ar.append(series[2])
                            prod_ar.append(series[7])
                            print(prod_ar)
    #                         res['title'].append(series[0])
    #                         res['entity'].append(series[4])

                    elif ('ram_range') in entity_dict['ram_category']:
                      a = ''.join(filter(lambda i: i.isdigit(), entity_dict['ram'][0]))
                      b = ''.join(filter(lambda i: i.isdigit(), entity_dict['ram'][1]))
                      price_cap_int_low = min(a,b)
                      price_cap_int_high = max(a,b)
                      entity_dict.pop('ram')
                      entity_dict.pop('ram_category')
                      # print(entity_dict)
                      dfprice = sql_search(entity_dict)
                      res['data']='The price of the requested products are shown below:-'
                      res['title']=[]
                      res['entity']=[]
                      # for row in dfprice:
                      for (row,series) in dfprice.iterrows():
                        price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[7]))
                        if price_cap_int_low <price_dat_int < price_cap_int_high:
            #               return(series[0],series[3])
                          res['title'].append(series[2])
                          res['entity'].append(series[7])










                if(flag>0):              
                    return res

                else:
                    dfprice = sql_search(entity_dict)
                    res['title']=[]
                    res["data"] = ' These are all the'
                    for key in entity_dict.keys():
                        res["data"] += ' ' + entity_dict[key][0]
                    res["data"] += ' in our catalogue. You can choose any one of them or even give me more specifications that you want so I can narrow down the results for you!'
            #           return(output, query_df.title, sep='\n')
                    for (row,series) in dfprice.iterrows():
                        res['title'].append(series[2])
                    return res


        # greetings intent
        elif intent == 'Greetings':
            res['data'] = 'Hello, I am Dorami. How can I help you?'
            return res

        
        # price_info intent
        elif intent == 'Price_info':
            dfnew = sql_search(entity_dict)
            res['data'] = 'The price of the requested products are shown below:-'
            res['title'] = []
            res['price'] = []
            for (row,series) in dfnew.iterrows():
                res['title'].append(series[2])
                res['price'].append(series[5])
            return res


        # bye intent
        elif intent == 'Bye':
            if len(past_intents) > 1 and past_intents[-2] == 'Buy_info':
                res['data'] = 'Great, I am glad you  like the products, do let me further technical specifications in which you want the product or you can just select any of the above products'
            else:
                replies = ['Goodbye and have a nice day!', 'Arigato Gozaimashita, Have a great day ahead!', 'Thankyou for shopping, have a fine day!']
                res['data'] = random.choice(replies)
            return res


        # yes intent
        elif intent == 'yes':
            res['data'] = 'I am happy that this is what you want!!'
            return res


        # no intent
        elif intent == 'no':
            replies = ['If you want any specific product you can provide me more details such as the specific brand name, product name or technical specifications too',
                      'I am sorry you did not like the products, you can chose to give me more details or contact Otsuka Shokai Sales for further inquiry',
                      'I am sorry you did not like the products I showed, can I know some more details about the kind of product you wanna see so I can show you better products?']
            res['data'] = random.choice(replies)
            return res
        

        # similar product intent
        elif intent == 'similar_product':
            similar_entity_dict = past_entities[-1].copy()
            brands = list(dflist.brand.unique())
            similar_entity_dict['brand'] = (random.choice(brands),)
            if 'product_name' in similar_entity_dict.keys():
                similar_entity_dict.pop('product_name')
            query_df = sql_search(similar_entity_dict)
            past_entities.append(similar_entity_dict)
            res['data'] = ('Dorami: These are some of the products similar to the previous one, have a look!', query_df.title)
            return res
        

        # comparison info intent
        elif intent == 'Comparison_info':
            if 'product_name' not in entity_dict.keys():
                res['data'] = 'What models do you want to compare?'
                return res
            res['data'] = comp_search(entity_dict)
            return res


        # spec info intent
        if intent == 'Specs_info':
          sql_search(entity_dict)
          chatbot()

 
    except:
        res['data'] = 'I am sorry I did not understand your question, could you please check if your question is correct otherwise try a different question or contact Otsuka Shokai Sales!'
        return res


# In[97]:


# this will be at the end of the notebook once ur done with testing
past_intents.clear()
past_entities.clear()


# In[98]:


# def test():
#     input_string = input('User: ')
#     res = chatbot(input_string)
#     return res


# In[102]:


# simulates frontend
# a = test()
# print(a)


# In[80]:


#from dorami import chatbot
# from flask import Flask, request
# app = Flask(__name__)
 


# In[81]:


#from dorami import chatbot
from flask import Flask, request
from flask_cors import CORS
app = Flask(_name_)
CORS(app)
 
# def _init_(self):
#     	self.chatbot = chatbot()


@app.route("/", methods=["POST"])
def hello_name():
	input_string = request.get_json()['input_string']
	res = chatbot(input_string) # res should be a dict
# 	res={
# 		"data": "These are the laptop",
# 		"title":["HP Pavilion Ryzen 5 Hexa Core 5600H - (8 GB/512 GB SSD/Windows 10/4 GB Graphics/NVIDIA GeForce GTX 1650/144 Hz) 15-ec2004AX Gaming Laptop  (15.6 inch, Shadow Black, 1.98 kg)","ASUS TUF Gaming A17 Ryzen 7 Octa Core 4800H - (16 GB/512 GB SSD/Windows 10 Home/4 GB Graphics/NVIDIA GeForce RTX 3050/144 Hz) FA706IC-HX003T Gaming Laptop  (17.3 inch, Graphite Black, 2.60 kg)"] ,
# 		"price" : ["100","200"] ,
# 		"img_url": ["https://rukminim1.flixcart.com/image/612/612/kbqu4cw0/computer/q/x/r/hp-original-imaftyzachgrav8f.jpeg?q=70","https://rukminim1.flixcart.com/image/612/612/l3rmzrk0/computer/z/2/c/-original-imagetjyhhtrtkdg.jpeg?q=70"] ,
# 		"title_url":["https://www.flipkart.com/hp-pavilion-ryzen-5-hexa-core-5600h-8-gb-512-gb-ssd-windows-10-4-graphics-nvidia-geforce-gtx-1650-144-hz-15-ec2004ax-gaming-laptop/p/itm98c94bbf9bc20?pid=COMG5GZXPWMGTNWS&lid=LSTCOMG5GZXPWMGTNWSQE9WVW&marketplace=FLIPKART&q=HP+Pavilion+Ryzen+5+Hexa+Core+5600H+-+%288+GB%2F512+GB+SSD%2F...%3B+ASUS+TUF+Gaming+A17+Ryzen+7+Octa+Core+4800H+-+%2816+GB%2F51...%3B+acer+Aspire+7+Core+i5+10th+Gen+-+%288+GB%2F512+GB+SSD%2FWindo...%3B+ASUS+TUF+Gaming+F15+Core+i5+10th+Gen+-+%288+GB%2F512+GB+SSD...&store=search.flipkart.com&srno=s_1_2&otracker=search&otracker1=search&fm=Search&iid=7788951f-4646-47a8-a746-6239f371432a.COMG5GZXPWMGTNWS.SEARCH&ppt=sp&ppn=sp&ssid=xcyp1j9xj40000001656243820753&qH=707549e6ce269adf",
# 						"https://www.flipkart.com/asus-tuf-gaming-a17-ryzen-7-octa-core-4800h-16-gb-512-gb-ssd-windows-10-home-4-graphics-nvidia-geforce-rtx-3050-144-hz-fa706ic-hx003t-laptop/p/itmccc79bbd17e07?pid=COMG8N5PFPCX9Z3J&lid=LSTCOMG8N5PFPCX9Z3JGKCEFC&marketplace=FLIPKART&cmpid=content_computer_15083003945_u_8965229628_gmc_pla&tgi=sem,1,G,11214002,u,,,556262839325,,,,c,,,,,,,&gclid=CjwKCAjwh-CVBhB8EiwAjFEPGcxYUKUL4RQPdprBhLDbEgKm7R14pHuY0o73wTzRXQ-amw29uNg3XBoC5mIQAvD_BwE"]		
# 	}
	
	return res


# In[82]:


if __name__ == '__main__':
	app.run(debug=True)


# In[83]:


# dflist


# In[84]:


# def test2():
#     user_query = input('User: ')
#     try:
#         # initializing the dictionary containing results to be returned to the frontend
#         res = dict()
        
#         # capturing intents and entities
#         intent, entity_dict = intents_ent(user_query)
#         print(intent, entity_dict)
        
        
#         # to capture missing product name in comparison info search
#         if past_intents != [] and past_intents[-1] == 'Comparison_info':
#             past_entity_dict = past_entities[-1]
#             past_entity_dict.update(entity_dict)
#             res['data'] = comp_search(past_entity_dict)
#             return res


#     #     # for missing_entity function
#     #     if a=="missing_ent":
#     #         if intent =="no":
#     #             #print('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title, sep='\n')
#     #             a="none"
#     #             pass
#     #         else:
#     #             past_entity_dict.update(entity_dict)
#     #             entity_dict=past_entity_dict.copy()

#         # just to check if code capturing right intents and entitites
#        # print(intent,entity_dict)

#     #     # when Dorami asks the user more details to narrow the search results
#     #     if past_intents != [] and past_intents[-1] == 'Buy_info':
#     #       #print(past_intents[-1] )
#     #       past_entity_dict = past_entities[-1]
#     #       past_entity_dict.update(entity_dict)
#     #       if 'brand' not in past_entity_dict.keys():
#     #            missing_entity('brand', past_entity_dict['product_type'][0], past_entity_dict)
#     # #           past_entity_dict.update(missing_entity_dict)
#     #       if 'product_name' not in past_entity_dict.keys():
#     #           missing_entity('product_name', past_entity_dict['product_type'][0], past_entity_dict)
#     # #           past_entity_dict.update(missing_entity_dict)
#     #       if 'colour' not in past_entity_dict.keys():
#     #           missing_entity('colour', past_entity_dict['product_type'][0], past_entity_dict)
#     # #           past_entity_dict.update(missing_entity_dict)

#     #       # appending spec_info intent at the end of intents list to avoid going into this if clause again in the next chatbot func iteration
#     #       past_intents.append('spec_info')
#     # #       return(past_entity_dict)
#     # #      print(past_entity_dict)
#     #       query_df = sql_search(past_entity_dict)
#     # #       return('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title, sep='\n')
#     #       p=('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title)
#     #       res.append(p)
#     #       return res

#         # when Dorami asks the user more details to narrow the search results
#         if past_intents != [] and (past_intents[-1] == 'Buy_info' or past_intents[-1] == 'similar_product') and intent not in ['Bye', 'yes', 'Q_A', 'Price_info', 'similar_product', 'Comparison_info']:
#             print(past_intents[-1] )
#             past_entity_dict = past_entities[-1]
#             past_entity_dict.update(entity_dict)
#             if 'product_type' not in past_entity_dict.keys():
#                 # for now since we are only dealing in laptops
#                 past_entity_dict['product_type'] = 'laptop'
#             if 'brand' not in past_entity_dict.keys():
#                 missing_entity_dict = missing_entity('brand', past_entity_dict['product_type'][0], past_entity_dict)
#                 past_entity_dict.update(missing_entity_dict)
#             if 'product_name' not in past_entity_dict.keys():
#                 missing_entity_dict = missing_entity('product_name', past_entity_dict['product_type'][0], past_entity_dict)
#                 past_entity_dict.update(missing_entity_dict)
#             if 'colour' not in past_entity_dict.keys():
#                 missing_entity_dict = missing_entity('colour', past_entity_dict['product_type'][0], past_entity_dict)
#                 past_entity_dict.update(missing_entity_dict)

#           # appending _ intent at the end of intents list to avoid going into this if clause again in the next chatbot func iteration
#             past_intents.append('_')
#             past_entities.append(past_entity_dict)
#             print(past_entity_dict)

#             query_df = sql_search(past_entity_dict)
#             print('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title, sep='\n')
#             chatbot()
            
# #         if past_intents != [] and (past_intents[-2] == 'Buy_info' or past_intents[-2] == 'similar_product') and :
# #             pass
            

#     #     # when Dorami asks the user more details to narrow the search results
#     #     elif len(past_intents) > 1 and (past_intents[-2] == 'Buy_info' or past_intents[-2] == 'similar_product') and intent not in ['Bye', 'yes']:
#     #       print(past_intents[-2] )
#     #       past_entity_dict = past_entities[-2]
#     #       past_entity_dict.update(entity_dict)
#     #       if 'product_type' not in past_entity_dict.keys():
#     #           past_entity_dict['product_type'] = 'laptop'
#     #       if 'brand' not in past_entity_dict.keys():
#     #           missing_entity_dict = missing_entity('brand', past_entity_dict['product_type'][0], past_entity_dict)
#     #           past_entity_dict.update(missing_entity_dict)
#     #       if 'product_name' not in past_entity_dict.keys():
#     #           missing_entity_dict = missing_entity('product_name', past_entity_dict['product_type'][0], past_entity_dict)
#     #           past_entity_dict.update(missing_entity_dict)
#     #       if 'colour' not in past_entity_dict.keys():
#     #           missing_entity_dict = missing_entity('colour', past_entity_dict['product_type'][0], past_entity_dict)
#     #           past_entity_dict.update(missing_entity_dict)

#     #       # appending spec_info intent at the end of intents list to avoid going into this if clause again in the next chatbot func iteration
#     #       past_intents.append('_')
#     #       past_entities.append(past_entity_dict)
#     #       print(past_entity_dict)

#     #       query_df = sql_search(past_entity_dict)
#     #       print('Dorami: These are the products I found based on your specifications, is this what you are looking for?', query_df.title, sep='\n')
#     #       chatbot()

#         # storing the intents and entities for use in above scenario and dialogue context tracking
#         past_intents.append(intent)
#         past_entities.append(entity_dict)


#         # q_a intent
#         if intent == 'Q_A':
#             res['data'] = Q_A(user_query)
#             return res


#         # buy_info
#         elif intent == 'Buy_info':
#             if 'laptop' in entity_dict['product_type']:
#                 # print(entity_dict)
#                 dict_ents_iterable = ['price','size','ram','weight']
#                 for entity in dict_ents_iterable:
#                     if str(str(entity)+'_category') in entity_dict:
#                         if str(str(entity)+'_above') in entity_dict[str(str(entity)+'_category')]:
#                           price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict[str(entity)][0]))
#         #                   print(price_cap_int)
#                           entity_dict.pop(str((entity)))
#                           entity_dict.pop(str(str(entity)+'_category'))
#                           # print(entity_dict)
#                           dfprice = sql_search(entity_dict)
#                           res['data']='The price of the requested products are shown below:-'
#                           res['title']=[]
#                           res['entity']=[] 
#                           # for row in dfprice:
#                           for (row,series) in dfprice.iterrows():
#                             price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[3+dict_ents_iterable.index(entity)]))
#         #                     print(price_dat_int)
#                             if price_dat_int > price_cap_int:
#                 #               return(series[0],series[3])

#                               res['title'].append(series[0])
#                               res['entity'].append(series[3+dict_ents_iterable.index(entity)])
#                             #  print(series[0],series[3+dict_ents_iterable.index(entity)])
#                             #chatbot()
#                         elif str(str(entity)+'_below') in entity_dict[str(str(entity)+'_category')]:
#                           price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict[str(entity)][0]))
#         #                   print(price_cap_int)
#                           entity_dict.pop(str((entity)))
#                           entity_dict.pop(str(str(entity)+'_category'))
#                           # print(entity_dict)
#                           dfprice = sql_search(entity_dict)
#                           # for row in dfprice:
#                           for (row,series) in dfprice.iterrows():
#                             price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[3+dict_ents_iterable.index(entity)]))
#         #                     print(price_dat_int)
#                             if price_dat_int < price_cap_int:
#                 #               return(series[0],series[3])
#                               res['title'].append(series[0])
#                               res['entity'].append(series[3+dict_ents_iterable.index(entity)])
#                         elif str(str(entity)+'_around') in entity_dict[str(str(entity)+'_category')]:
#                           t = 0
#                           price_cap_int = ''.join(filter(lambda i: i.isdigit(), entity_dict[str(entity)][0]))
#                           entity_dict.pop(str((entity)))
#                           entity_dict.pop(str(str(entity)+'_category'))
#                           # print(entity_dict)
#                           dfprice = sql_search(entity_dict)
#                           prod_ar = []
#                           for (row,series) in dfprice.iterrows():
#                             price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[3+dict_ents_iterable.index(entity)]))
#                             if abs(int(price_dat_int) - int(price_cap_int))<abs(int(t)-int(price_cap_int)):
#                               t = price_dat_int
#                           for (row,series) in dfprice.iterrows():
#                             price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[3+dict_ents_iterable.index(entity)]))
#                             if t == price_dat_int:
#                               res['title'].append(series[0])
#                               res['entity'].append(series[3+dict_ents_iterable.index(entity)])

#                         elif str(str(entity)+'_range') in entity_dict[str(str(entity)+'_category')]:
#                           a = ''.join(filter(lambda i: i.isdigit(), entity_dict[str(entity)][0]))
#                           b = ''.join(filter(lambda i: i.isdigit(), entity_dict[str(entity)][1]))
#                           price_cap_int_low = min(a,b)
#                           price_cap_int_high = max(a,b)
#                           entity_dict.pop(str((entity)))
#                           entity_dict.pop(str(str(entity)+'_category'))
#                           # print(entity_dict)
#                           dfprice = sql_search(entity_dict)
#                           # for row in dfprice:
#                           for (row,series) in dfprice.iterrows():
#                             price_dat_int = ''.join(filter(lambda i: i.isdigit(), series[3+dict_ents_iterable.index(entity)]))
#                             if price_cap_int_low <price_dat_int < price_cap_int_high:
#                 #               return(series[0],series[3])
#                               res['title'].append(series[0])
#                               res['entity'].append(series[3+dict_ents_iterable.index(entity)])


#             else:
#                 query_df = sql_search(entity_dict)
#                 res["data"] = ' These are all the'
#                 for key in entity_dict.keys():
#                     res["data"] += ' ' + entity_dict[key][0]
#                 res["data"] += ' in our catalogue. You can choose any one of them or even give me more specifications that you want so I can narrow down the results for you!'
#         #           return(output, query_df.title, sep='\n')
#             for (row,series) in dfprice.iterrows():
#                 res['title'].append(series[0])
#             return res


#         # greetings intent
#         elif intent == 'Greetings':
#             res['data'] = 'Hello, I am Dorami. How can I help you?'
#             return res

        
#         # price_info intent
#         elif intent == 'Price_info':
#             dfnew = sql_search(entity_dict)
#             res['data'] = 'The price of the requested products are shown below:-'
#             res['title'] = []
#             res['price'] = []
#             for (row,series) in dfnew.iterrows():
#                 res['title'].append(series[0])
#                 res['price'].append(series[3])
#             return res


#         # bye intent
#         elif intent == 'Bye':
#             if len(past_intents) > 1 and past_intents[-2] == 'Buy_info':
#                 res['data'] = 'Great, I am glad you  like the products, do let me further technical specifications in which you want the product or you can just select any of the above products'
#             else:
#                 replies = ['Goodbye and have a nice day!', 'Arigato Gozaimashita, Have a great day ahead!', 'Thankyou for shopping, have a fine day!']
#                 res['data'] = random.choice(replies)
#             return res


#         # yes intent
#         elif intent == 'yes':
#             res['data'] = 'I am happy that this is what you want!!'
#             return res


#         # no intent
#         elif intent == 'no':
#             replies = ['If you want any specific product you can provide me more details such as the specific brand name, product name or technical specifications too',
#                       'I am sorry you did not like the products, you can chose to give me more details or contact Otsuka Shokai Sales for further inquiry',
#                       'I am sorry you did not like the products I showed, can I know some more details about the kind of product you wanna see so I can show you better products?']
#             res['data'] = random.choice(replies)
#             return res
        

#         # similar product intent
#         elif intent == 'similar_product':
#             similar_entity_dict = past_entities[-1].copy()
#             brands = list(dflist.brand.unique())
#             similar_entity_dict['brand'] = (random.choice(brands),)
#             if 'product_name' in similar_entity_dict.keys():
#                 similar_entity_dict.pop('product_name')
#             query_df = sql_search(similar_entity_dict)
#             past_entities.append(similar_entity_dict)
#             res['data'] = ('Dorami: These are some of the products similar to the previous one, have a look!', query_df.title)
#             return res
        

#         # comparison info intent
#         elif intent == 'Comparison_info':
#             if 'product_name' not in entity_dict.keys():
#                 res['data'] = 'What models do you want to compare?'
#                 return res
#             res['data'] = comp_search(entity_dict)
#             return res


#         # spec info intent
#         if intent == 'Specs_info':
#           sql_search(entity_dict)
#           chatbot()

 
#     except:
#         res['data'] = 'I am sorry I did not understand your question, could you please check if your question is correct otherwise try a different question or contact Otsuka Shokai Sales!'
#         return res
    


# In[85]:


# b = test2()
# print(b)


# In[86]:


# # this will be at the end of the notebook once ur done with testing
# past_intents.clear()
# past_entities.clear()


# In[ ]:




