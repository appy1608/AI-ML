#!/usr/bin/env python
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:

#Loading the data into Pandas dataframe

df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')
df.head()

# In[8]:

df.info()

# In[9]:

#For understanding the distribution better
sns.countplot(df.v1)   
plt.xlabel('Label')
plt.title('Number of ham and spam messages')
plt.show()
# In[70]:

X = df.v2     #Create input and output vectors.
Y = df.v1
#df['v1'].unique()
le = LabelEncoder()    
Y = le.fit_transform(Y)  #Process the labels.
Y = Y.reshape(-1,1)
print(Y)

# In[72]:

#Splitting into training and testing data.

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)
# print(X_train[31])


# In[73]:

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words) #Tokenize the data 
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train) # Convert the text to sequences.
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)  #Add padding to ensure that all the sequences have the same shape.

# In[77]:

#print(sequences[0])
#print(sequences_matrix)

# In[78]:

#Defining the RNN structure.

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

# In[83]:

#Calling the function and compile the model.

model= RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

# In[85]:

#Fitting on the training data.

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

# In[86]:

#The model performs well on the validation set and this configuration is chosen as the final model.

#Process the test set data.

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

# In[87]:

#Evaluate the model on the test set.

accr = model.evaluate(test_sequences_matrix,Y_test) 

# In[88]:

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
