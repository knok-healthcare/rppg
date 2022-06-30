from keras.models import Sequential
from keras.layers import Conv2D
import directories
from src.data_processing.model_input_preparation import read_fft_df
import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop,SGD,Adam


def cnn(input, parts):

    # input shape
    colors = 3
    fft_len = 474


    model = Sequential()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(fft_len, parts, colors)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(550, activation="relu"),  # Adding the Hidden layer
        tf.keras.layers.Dropout(0.1, seed=2019),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dropout(0.3, seed=2019),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dropout(0.4, seed=2019),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dropout(0.2, seed=2019),
        tf.keras.layers.Dense(5, activation="softmax")  # Adding the Output Layer
    ])
    # compile the keras model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)
    print(model.get_weights())
    # apply filter to input data
    yhat = model.predict(data)
    for r in range(yhat.shape[1]):
        # print each column in the row
        print([yhat[0, r, c, 0] for c in range(yhat.shape[2])])


if __name__ == "__main__":
    directories.change_dir_to_main()
    input_dir = 'data/prepared_model_input/fft/BPM80T15'

    input = read_fft_df(input_dir)