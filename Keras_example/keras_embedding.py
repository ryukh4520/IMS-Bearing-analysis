
# 20Newsgroup dataset classification
# using GLove word embedding model

# conv1d

from __future__ import print_function

import os
import sys
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPool1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


# Create index based on GLove. Matching embedding vector per word

# path
Base_DIR = '/home/gon/Desktop/KwangHyun/code_practice/Keras_example/dataset'
GLOVE_DIR = os.path.join(Base_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(Base_DIR, '20_newsgroup')


# setting parameters
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORD = 20000
Embedding_DIM = 100
VALIDATION_SPLIT = 0.2


print('Indexing word vectors.')

# create embedding index
embedding_index = {}
with open(os.path.join(Base_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        # maxsplit > Specify number of word
        word, coefs = line.split(maxsplit=1)
        # numpy fromstring > string to array
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embedding_index[word] = coefs

        # the / -0.0123 , ...
        # word / coefs

print('Found %s word vectors.' % len(embedding_index))

# new data sample , label matching
texts = []
labels_index = {}
labels = []


for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)

    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id

        for fname in sorted(os.listdir(path)):
            if fname.isdigit():

                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else{'encoding' : 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n') # skip header, find return (index val / -1)

                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts. ' % len(texts))

tokenizer = Tokenizer(num_words=MAX_NUM_WORD)
tokenizer.fit_on_texts(texts)
sequence = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' %len(word_index))

# pad_sequence > for Regular intervals (MAX_sequence_length)
data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

# one-hot encoding
labels = to_categorical(np.asarray(labels))

print('shape of data tensor:', data.shape)
print('shape of label tensor:', labels.shape)


indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')


num_words = min(MAX_NUM_WORD, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, Embedding_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORD:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(num_words,
                            Embedding_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model')


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32')
embedding_sequence = embedding_layer(sequence_input)

# model setting & parameter

x = Conv1D(128, 5, activation='relu')(embedding_sequence)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPool1D()(x)
x = Dense(128, activation='relu')(x)

preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.summary()

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val)
          )
