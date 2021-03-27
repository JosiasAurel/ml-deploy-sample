import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
  keras.layers.Dense(1, activation="relu", input_shape=[1])
])

sample_input = (1, 2, 3, 4, 5, 6)
sample_output = (3, 5, 7, 9, 11, 13)

model.compile(optimizer="sgd", loss="mean_squared_error")

# train model
model.fit(sample_input, sample_output, epochs=500)

# model.save(filepath="./")

print(model.predict([8]))