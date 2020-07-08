#single_layer
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt



#set_training_data
celsius_q   =   np.array([-40., -10, 0, 8, 15, 22, 38], dtype=float)
fahreheit_a =   np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)


for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahreheit".format(c,fahreheit_a[i]))


#create_model
    #layer l0
l0 = tf.keras.layers.Dense(units=1, input_shape =[1])

model = tf.keras.Sequential([l0])



#compile_model

model.compile(loss='mean_squared_error',
              optimizer = tf.keras.optimizers.Adam(0.1))
        ##MSE - OPTIMIZER_is_used_for_getting_better_result


#train_model
history = model.fit(celsius_q, fahreheit_a, epochs=500, verbose=False)
print("Finished training the model")


#print_graph

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.savefig('tryhard_image.png')
plt.show()

#check_specific_value
print(model.predict([100.0]))
print("These are the layer variables: {}".format(l0.get_weights()))
    # F = C * 1.8 +32


