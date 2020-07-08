import os
import glob
import itertools
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers

tf.enable_eager_execution()
start = time.time()

normal_path = '/home/gon/Desktop/0622/dataset/1.normal'
inner_path = '/home/gon/Desktop/0622/dataset/2.inner'
outer_path = '/home/gon/Desktop/0622/dataset/3.outer'
roller_path = '/home/gon/Desktop/0622/dataset/4.roller'


# filelist count
normal_files = glob.glob(normal_path + "/*")
print("Normal files count: ", len(normal_files))

inner_files = glob.glob(inner_path + "/*")
print("Inner files count: ", len(inner_files))

outer_files = glob.glob(outer_path + "/*")
print("Outer files count: ", len(outer_files))

roller_files = glob.glob(roller_path + "/*")
print("Roller files count: ", len(roller_files))


# train-test data split
normal_files_train, normal_files_val = train_test_split(normal_files[:650], test_size=100, random_state=32)
inner_files_train, inner_files_val = train_test_split(inner_files[:650], test_size=100, random_state=323)
outer_files_train, outer_files_val = train_test_split(outer_files[:650], test_size=100, random_state=123)
roller_files_train, roller_files_val = train_test_split(roller_files[:650], test_size=100, random_state=285)

print("Normal data split :", len(normal_files_train), len(normal_files_val))
print("Inner data split :", len(inner_files_train), len(inner_files_val))
print("Outer data split :", len(outer_files_train), len(outer_files_val))
print("Roller data split :", len(roller_files_train), len(roller_files_val))
print("\n")


train_files = normal_files_train + inner_files_train + outer_files_train + roller_files_train
val_files = normal_files_val + inner_files_val + outer_files_val + roller_files_val
test_files = normal_files[650:750] + inner_files[650:750] + outer_files[650:750] + roller_files[650:750]

print("Total len of Train files : ", len(train_files))
print("Total len of Validation files : ", len(val_files))
print("Total len of Test files : ", len(test_files))
print("\n")

np.random.shuffle(train_files)
np.random.shuffle(val_files)


# data_generate, considering chunk_files truncated by batch_size
def tf_d_generator(files, batch_size=4):
    i = 0
    while True:
        if i * batch_size >= len(files):
            i = 0
            np.random.shuffle(files)

        else:
            file_chunk = files[i*batch_size:(i+1)*batch_size]
            data = []
            labels = []
            patterns = tf.constant([".*(normal)", ".*(inner)", ".*(outer)", ".*(roller)"])

            for file in file_chunk:
                temp = pd.read_csv(open(file, 'r'), sep="\s+", header=None)
                fault_columns = [0, 4, 2, 6] # [normal, inner, outer, roller]
                num = np.int(np.floor(len(temp[0])/1024)) # all columns have same number of entries
                j = 0
                for pattern in patterns:
                    if re.match(pattern.numpy(), tf.constant(file).numpy()):
                        labels = labels + list(np.repeat(j, num))
                        column_number = fault_columns[j]
                        break
                    j = j + 1
                data = data + list(temp[column_number][0:num*1024].values.reshape(num, 32, 32, 1))

            data = np.asarray(data).reshape(-1, 32, 32, 1)
            labels = np.asarray(labels)

            index = np.random.permutation(len(data))
            data, labels = data[index], labels[index]

            yield data, labels
            i = i + 1


# model construction
batch_size = 50

train_dataset = tf.data.Dataset.from_generator(tf_d_generator, args=[train_files, batch_size],
                                               output_shapes=((None, 32, 32, 1), (None, )), output_types=(tf.float32, tf.float32))

val_dataset = tf.data.Dataset.from_generator(tf_d_generator, args=[val_files, batch_size],
                                             output_shapes=((None, 32, 32, 1), (None,)), output_types=(tf.float32, tf.float32))

test_dataset = tf.data.Dataset.from_generator(tf_d_generator, args=[test_files, batch_size],
                                             output_shapes=((None, 32, 32, 1), (None,)), output_types=(tf.float32, tf.float32))



# prefetch - 전처리와 훈련 스텝의 모델 실행을 오버랩한다, Autotune은 자동으로 tf runtime이 이를 동적으로 조정해준다
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


"""여기서 문제 발생, keras -> model fit할때, 배열 형태로 데이터 입력 받으나, """
"""
# model_keras
model = Sequential()
model.add(layers.Conv2D(32, 5, activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(16, 5, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()


"""
model = Sequential([
    layers.Conv2D(32, 5, activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(16, 5, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.summary()

# model parameter info, ceil > 올림
epoch_step = np.ceil(len(train_files)/batch_size)
val_step = np.ceil(len(val_files)/batch_size)
test_step = np.ceil(len(test_files)/batch_size)

epoch_step = int(epoch_step)
val_step = int(val_step)
test_step = int(test_step)

print('each steps for epoch\n Train : {', epoch_step, '}, Validation : {', val_step, '}, Test : {', test_step, '}')
print("\n")

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=["accuracy"])


model.fit(train_dataset, validation_data= val_dataset,
          steps_per_epoch=epoch_step,
          validation_steps=val_step,
          epochs=10
          )


test_loss, test_acc = model.evaluate(test_dataset, steps= test_step)
print("time :", time.time() - start)
#model.save("IMS_final.h5")


