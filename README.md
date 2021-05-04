# Seq2seq Model 

### Goal:

The goal of the project is to build sequence-to-sequence model, which translates english sentences into german. For this task I will use dataset from deu.txt file, where one record is english sentence and its german translation

### Seq2seq idea:

Sometimes input sequence is equal to output sequence. In this case we can use simple LSTM architecture for a task. But in some cases the length of input and output differs. A good example is language translation. For example an english sentence 'I go home' and its german translation 'Ich gehe nach Hause' have different length. For this purpose we will use encoder - decoder architecture.

a)Encoder: basically what encoder does is it learns hidden features of the input(in our case English sentences) and return hidden state and cell state.
b)Decoder: it takes hidden and cell state from the Encoder as initial states and predicts and output at each time step. 

During training we use teacher forcing: the input at each time step is actual output and not the predicted output from the last time step.

### Version of Tensorflow:

Implementation was done using Tensorflow 2.4 version.

###

For the proper functionality in google colab the following commands must be executed:

1. Kaggle package:
```
from google.colab import files
!pip install -q kaggle
uploaded = files.upload()
```

2. Downloading pretrained Embedding:

```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d takuok/glove840b300dtxt
!unzip glove840b300dtxt.zip
```
