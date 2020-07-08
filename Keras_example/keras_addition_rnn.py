# -*- coding: utf-8 -*-

# Addition problem
# Input = string , str + str = result?

from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range


class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars)) #0123456789+
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars)) #

        # initialize, generate indices // dict - dictionary

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

        # encode result = C, num_rows -> rows returned in one-hot encoding, keep the # rows of the data

    def decode(self, x, calc_argmax=True):
        # Decode given vector(2D array) to their character output
        # calc_argmax = find the character index with maximum probability
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)
                # printing like a b c ...


class colors:
    # color print on terminal, ANSI Escape Sequence
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# parameter
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

# ex 365 + 789 = 1 (input string)
MAXLEN = DIGITS + 1 + DIGITS


chars = '0123456789+ '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()

print("generating Data..")
# Character table + input

# create data
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('012345678'))
                for i in range(np.random.randint(1, DIGITS + 1))))
    # 0123456789 to number

    # generate a,b for addition
    a, b = f(), f()

    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)

    # pad the data with spaces such that it is always MAXLEN
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))

    ans = str(a + b)

    ans += ' ' * (DIGITS + 1 - len(ans))
    # answer maxlen(DIGITS + 1)
    if REVERSE:
        # 12+345 -> 543 + 21
        query = query[::-1]

    questions.append(query)
    expected.append(ans)


print('Total addition questions :', len(questions))
print('Vectorization')

# for one hot encoding
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype = np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype = np.bool)

for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)


# Shuffle x,y in unison as the later parts of x will almost all be larger digits
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]


split_at = len(x) - len(y) // 10

(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training data:')
print(x_train.shape)
print(y_train.shape)


print('Validation data: ')
print(x_val.shape)
print(y_val.shape)


RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

# model structure
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
# 999 + 999 = 1998


for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE,return_sequences=True))

# applied independently Dense layer to n_timestep
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# validation
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))

    # 10 sample, error visualization
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
            print(guess)
