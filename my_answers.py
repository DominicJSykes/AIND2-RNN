import numpy as np
import string

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    #get P (length of series) - T (window_size) pairs
    for i in range(len(series) - window_size):
        
        #container for each sequence created
        x = []
        
        #get a T (window_size) length sequence
        for j in range(window_size):
            x.append(series[i+j])
            
        #append each sequence - output pair to the containers for input/output pairs
        X.append(x)
        y.append(series[i+window_size])
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    #build a model with a LSTM hidden layer and a regression prediction layer
    model = Sequential()
    model.add(LSTM(5,input_shape = (window_size,1)))
    model.add(Dense(1))
    
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?',' ']
    
    #list unique characters in text
    unique_chars = sorted(list(set(text)))
    
    #iterate over the unique characters in the text
    for i in unique_chars:
        
        #if character isn't in the accepted punctuation or the lowercase alphabet then replace with a space. 
        if i not in punctuation and i not in string.ascii_lowercase:
            text = text.replace(i,' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    #get (P (length of text) - T (window_size)) / M (step_size) pairs
    for i in range(int((len(text)-window_size)/step_size)+1):
        
        #string to contain each input sequence
        x = ""
        
        #get a window_size sequence
        for j in range(window_size):
            x = x + text[i*step_size+j]
        
        #append each sequence - output pair to the containers for input/output pairs
        inputs.append(x)
        outputs.append(text[i*step_size + window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    #build a model with a LSTM hidden layer and a softmax classification layer
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    
    return model
