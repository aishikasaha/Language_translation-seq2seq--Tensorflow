import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy.random import choice
import random
import re
from itertools import repeat
import tensorflow as tf
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
import string
import os
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

input_text = [] #original language
target_text_input = []#target language
target_text_output = []# target language offset by 1



def subst(phrase):
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


NUM_LINES = 100000
with open('deu.txt','r') as file:
    i = 0
    for line in file:
        i += 1
        if i > NUM_LINES:
            break

        if '\t' not in line:
            continue
        input_t, target_t,_ = line.split('\t')


        # input and output preprocessing
        input_t = subst(input_t)
        input_t = re.sub(r"([?.!,¿])", r" \1 ", input_t)
        input_t = re.sub(r'[" "]+', " ", input_t)
        input_t = re.sub(r"[^a-zA-Z?.!,¿]+", " ", input_t)
        input_t = input_t.strip().lower()

        target_t = re.sub(r"([?.!,¿])", r" \1 ", target_t)
        target_t = re.sub(r'[" "]+', " ", target_t)
        target_t = re.sub(r"[^a-zA-ZäöüÄÖÜß?.!,¿]+", " ", target_t)
        target_t = target_t.strip().lower()


        target_text_i = '<sos> ' + target_t  
        target_text_o = target_t + ' <eos>'


        input_text.append(input_t)
        target_text_input.append(target_text_i)
        target_text_output.append(target_text_o)





##tokenizer input:

#tokenizer
n_words = 20000
tokenizer_input = Tokenizer(oov_token='UNK',num_words = n_words,filters = '')

tokenizer_input.fit_on_texts(input_text)
word2index_input = tokenizer_input.word_index
index2word_input = tokenizer_input.index_word


## tokenizer output:
#tokenizer
tokenizer_output = Tokenizer(oov_token='UNK',num_words = n_words,filters = '')

tokenizer_output.fit_on_texts(target_text_output + target_text_input)
word2index_output = tokenizer_output.word_index
index2word_output = tokenizer_output.index_word

##words into numbers
input_seq = tokenizer_input.texts_to_sequences(input_text)
target_seq_input = tokenizer_output.texts_to_sequences(target_text_input)
target_seq_output = tokenizer_output.texts_to_sequences(target_text_output)

#padding
all_seq_target = target_seq_input  + target_seq_output
max_seq_len_input = max(len(x) for x in input_seq)
max_seq_len_output= max(len(x) for x in all_seq_target)




encoder_input = pad_sequences(input_seq,padding = 'post',maxlen = max_seq_len_input)
decoder_input = pad_sequences(target_seq_input,padding = 'post',maxlen = max_seq_len_output)
decoder_output = pad_sequences(target_seq_output,padding = 'post',maxlen = max_seq_len_output)


#load pretrained glove embeddings
glove_embedding = dict()

with open('glove.840B.300d.txt',encoding='utf8') as file:
    for line in file:
        values = line.split()
        word = ''.join(values[:-300])
        weights = np.asarray(values[-300:], dtype='float32')
        glove_embedding[word] = weights 

#Parameters:
batch_size = 32
dropout_rate = 0.3
embed_size = 300
encoder_units = 512
decoder_units = 512


#creating matrix from pretrained embeddings
embedding_matrix = np.random.normal(0,1,(n_words, embed_size))

for word, index in word2index_input.items():
    if index >= n_words:
        break
    if word in glove_embedding.keys():
        embedding_matrix[index] = glove_embedding[word]




class Encoder(Model):
    def __init__(self,embed_size,encoder_units,total_words,dropout_rate,embedding_matrix,batch_size):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.encoder_units = encoder_units
        self.total_words = total_words
        self.dropout_rate = dropout_rate
        self.embedding_matrix = embedding_matrix
        self.batch_size = batch_size

        #Encoder
        self.encoder_emb = Embedding(batch_input_shape=[self.batch_size , None],input_dim = self.total_words,output_dim = self.embed_size,weights=[self.embedding_matrix],trainable = True)
        self.lstm_encoder = LSTM(units = self.encoder_units,recurrent_dropout = self.dropout_rate,return_state = True,recurrent_initializer='glorot_uniform')

    def call(self,inputs_enc):
        x_enc = self.encoder_emb(inputs_enc)#Turns positive integers (indexes) into dense vectors of fixed size. E.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        x_enc,hidden,cell = self.lstm_encoder(x_enc)
        return hidden,cell

    
class Decoder(Model):
    def __init__(self,embed_size,total_words,dropout_rate,num_classes,batch_size,decoder_units):
        super(Decoder,self).__init__()
        self.embed_size = embed_size
        self.total_words = total_words
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decoder_units = decoder_units

        #Decoder 
        self.decoder_emb = Embedding(batch_input_shape=[self.batch_size , None],input_dim = self.total_words,output_dim = self.embed_size,trainable = True)
        self.lstm_decoder = LSTM(units = self.decoder_units,recurrent_dropout = self.dropout_rate,return_sequences = True,return_state = True,recurrent_initializer='glorot_uniform')
        self.Output = TimeDistributed(Dense(units = self.num_classes,activation = None))


    def call(self,inputs_dec):
        inputs,hidden,cell = inputs_dec 
        x_dec = self.decoder_emb(inputs)
        x_dec,hidden_dec,cell_dec = self.lstm_decoder(x_dec,initial_state = [hidden,cell])
        x_dec = self.Output(x_dec)
        return x_dec,hidden_dec,cell_dec


    
dataset_train = tf.data.Dataset.from_tensor_slices((encoder_input,decoder_input,decoder_output))
dataset_train = dataset_train.shuffle(buffer_size = 1024).batch(batch_size)

encoder = Encoder(batch_size = batch_size,embed_size = embed_size,total_words = n_words , dropout_rate = dropout_rate,embedding_matrix = embedding_matrix,encoder_units = encoder_units)
decoder = Decoder(batch_size = batch_size,embed_size = embed_size,total_words = n_words , dropout_rate = dropout_rate,num_classes = n_words,decoder_units = decoder_units)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy =tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


@tf.function
def training(X_1,X_2,y):
    #predictions
    with tf.GradientTape() as tape:
        hidden,cell =  encoder(X_1)
        predictions,_,_ = decoder((X_2,hidden,cell))
        loss = loss_object(y,predictions)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss,variables)
    optimizer.apply_gradients(zip(gradients,variables))
    train_loss(loss)
    train_accuracy(y,predictions)
    return hidden,cell
	
	
	
EPOCHS = 2

for epoch in range(EPOCHS):
    for X_1,X_2,y in dataset_train:
       hidden,cell = training(X_1,X_2,y)
    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch+1,train_loss.result(),train_accuracy.result()*100))
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()



def translator(sentence):
    result = []

    sentence = subst(sentence)
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip().lower()

    inputs = tokenizer_input.texts_to_sequences([sentence])
    inputs = pad_sequences(inputs,padding = 'post',maxlen = max_seq_len_input)
    inputs = tf.convert_to_tensor(inputs)
    
    hidden,cell = encoder(inputs)
    states_values = [hidden,cell]
    
    decoder_input = tf.expand_dims([word2index_output['<sos>']],0)


    for i in range(max_seq_len_output):
        decoder_output,hidden,cell = decoder.predict([decoder_input] + states_values)
        decoder_output = tf.squeeze(decoder_output)
        decoder_output = tf.nn.softmax(decoder_output).numpy().astype('float64')
        decoder_output = decoder_output / np.sum(decoder_output)
        indx = np.random.multinomial(n = 100,pvals = decoder_output,size = 1)
        ind = np.argmax(indx)
        if ind == word2index_output['<eos>']:
            break
        word = index2word_output[ind]

        #updating variables:
        states_values = [hidden,cell]
        decoder_input = tf.expand_dims([ind],0)

        result.append(word) 

    
    return ' '.join(result)



translator('I like you very much!')
translator('I want to watch a movie!')
translator('I like to play football')
translator('I want to go with you to the cinema')




