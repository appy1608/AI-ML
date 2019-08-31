#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)


# In[42]:


# load the dataset but only keep the top5000 words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# In[43]:


print(X_train[1])
print(len(X_train[1]))


# In[44]:


# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# In[48]:


print(len(X_train[1]))
print(y_train[:5])


# In[49]:


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))  #Embedded Layer(32 length vector)
model.add(LSTM(100))                                                                     #LSTM Layer(100 memory units)
model.add(Dense(1, activation='sigmoid'))                                                # Dense Output Layer (single neuron, AF- Sigmoid Function)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       # Adam optimization algorithm is used
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


# In[51]:


#LSTM Sequence Classification with Dropout
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       # Adam optimization algorithm is used
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


# In[52]:


# Final evaluation of the model on unseen reviews
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




