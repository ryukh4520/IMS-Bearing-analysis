import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(X_train0, Y_train0), (X_test0, Y_test0) = mnist.load_data()

# width / height
plt.figure(figsize=(6, 1))

# Create sample data window
for i in range(36):
    plt.subplot(3, 12, i+1)
    plt.imshow(X_train0[i], cmap="gray")
    plt.axis("off")

# plt.show()

# print data shape / type
print(X_train0.shape, X_train0.dtype)
print(Y_train0.shape, Y_train0.dtype)
print(X_test0.shape, X_test0.dtype)
print(Y_test0.shape, Y_test0.dtype)


X_train = X_train0.reshape(28, 28, 1).astype('float32')
X_test = X_test0.reshape(28, 28, 1).astype('float32')

print(X_train.shape, X_train.dtype)

# answer_data is digitization with labels
Y_train0[:5]

Y_train = tf.keras.utils.to_categorical(Y_train0, 10)
Y_test = tf.keras.utils.to_categorical(Y_test0, 10)
Y_train[:5]


np.random.seed(0)


model = tf.keras.models.Sequential()

# The first layer must set the input size
model.add(tf.keras.layers.Dense(15, activation="sigmoid", input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Dense(10, activation="sigmoid"))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.2), loss='mean_squared_error', metrics=["accuracy"])

model.summary()

hist = model.fit(X_train, Y_train, epochs=10, batch_size=100, validation_data=(X_test, Y_test))

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'])
plt.title("훈련 중 비용함수 그래프")
plt.ylabel("비용함수 값")

plt.subplot(1, 2, 2)
plt.title("훈련 중 성능지표 그래프")
plt.ylabel("성능지표 값")

plt.plot(hist.history['accuracy'], 'b-', label="Training performance")
plt.plot(hist.history['val_accuracy'], 'r:', label="Validation performance")




