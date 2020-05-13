import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional

df = pd.read_csv('../data/sales_train_validation.csv')
# df_sub = pd.read_csv('../data/sample_submission.csv')
#
# df_merge = pd.merge(df_sub, df, how='inner', on='id')


df_train = df[[col for col in df if col.startswith('d_') and int(col.replace('d_','')) > 1823]]
values = df_train.values

batch_size = 100
values_len = len(values)

n_steps, n_features, output_size = 90, 1, 30

model_path = '../checkpoints/modelBS'
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
    model.load_weights(model_path)
    return model

model = get_model()
print(values.shape)
values = values.reshape((values.shape[0], values.shape[1], n_features))
print(values.shape)
prediction = model.predict([values])
print(prediction.shape)

validation = {'id': df['id']}
for i in range(prediction.shape[1]-2):
    validation['F'+str(i+1)] = prediction[:, i]


df_validation = pd.DataFrame(validation)


df_train = df[[col for col in df if col.startswith('d_') and int(col.replace('d_','')) > 1853]]
values = df_train.values

evaluation = np.concatenate((values, prediction), axis=1)
evaluation = evaluation.reshape((evaluation.shape[0], evaluation.shape[1], n_features))
evaluation_prediction = model.predict([evaluation])
print(evaluation.shape)

evaluation = {'id': df['id'].str.replace('_validation', '_evaluation')}
for i in range(evaluation_prediction.shape[1]-2):
    evaluation['F'+str(i+1)] = evaluation_prediction[:, i]

df_evaluation = pd.DataFrame(evaluation)

df = pd.concat([df_validation, df_evaluation])
df.to_csv('../data/submission.csv', index=False)

