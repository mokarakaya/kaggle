import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional


model_path = '../checkpoints/modelBS'

df = pd.read_csv('../data/sales_train_validation.csv')
df_train = df[[col for col in df if col.startswith('d_')]]
values = df_train.values

batch_size = 100
values_len = len(values)

n_steps, n_features, output_size = 90, 1, 30


def get_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l1_l2()), input_shape=(n_steps, n_features)))
    model.add(Bidirectional(LSTM(2, kernel_regularizer=tf.keras.regularizers.l1_l2())))
    # model = Sequential()
    # model.add(LSTM(20,
    #                activation='tanh',
    #                input_shape=(n_steps, n_features),
    #                kernel_regularizer=tf.keras.regularizers.l1_l2()))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    model.compile(optimizer='adam', loss='mse')
    # model.load_weights(model_path)
    return model

model = get_model()

# fit model

days_length = values.shape[1]
window_slide = 30
window_start = 0

best_loss = 1000
validation = 30
X_validation = values[:, validation: validation + n_steps]
X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], n_features))
y_validation = values[:, validation + n_steps:validation + n_steps + output_size]
for i in range(2):
    total_range = range(window_start, days_length - output_size - n_steps, window_slide)
    for j in total_range:
        if j == validation:
            continue
        X = values[:, j: j + n_steps]
        y = values[:, j + n_steps:j + n_steps + output_size]
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        try:
            history = model.fit(X, y, epochs=1, verbose=1, batch_size=32, validation_data=(X_validation, y_validation))
            print(history)
            loss = min(history.history['val_loss'])
            if best_loss > loss:
                print('best loss ', best_loss)
                best_loss = loss
                model.save_weights(model_path)
        except ValueError:
            print('value error!!!! ', j)

        print(j, '/', total_range)
