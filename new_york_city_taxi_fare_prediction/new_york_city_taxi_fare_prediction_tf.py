import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.neural_network import MLPRegressor
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

chunk_size = 5000

clf = MLPRegressor(verbose=True)
nyc = (-74.0063889, 40.7141667)
jfk = (-73.7822222222, 40.6441666667)

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


feature_classes = ["distance_miles",
                   "year", "hour", 'weekday', 'passenger_count']



def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.tanh(layer_1)

    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.tanh(layer_2)

    W_O = tf.Variable(tf.random_uniform([10,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)

    return output,W_O

print(datetime.datetime.now())
xs = tf.placeholder("float")
ys = tf.placeholder("float")

output, W_O = neural_net_model(xs, len(feature_classes))

cost = tf.reduce_mean(tf.square(output - ys))
# train = tf.train.AdamOptimizer(0.05).minimize(cost)
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for train_df in pd.read_csv('data/train.csv', chunksize=chunk_size, parse_dates=['pickup_datetime']):

        train_df.dropna(how='any', inplace=True)

        train_df['distance_miles'] = distance(train_df.pickup_latitude, train_df.pickup_longitude,
                                              train_df.dropoff_latitude, train_df.dropoff_longitude).round(3)
        # train_df['distance_to_center'] = distance(nyc[1], nyc[0], train_df.pickup_latitude, train_df.pickup_longitude)
        # train_df['distance_to_center_drop_off'] = distance(nyc[1], nyc[0], train_df.dropoff_latitude, train_df.dropoff_longitude)

        # train_df['distance_to_jfk'] = distance(jfk[1], jfk[0], train_df.pickup_latitude, train_df.pickup_longitude)
        # train_df['distance_to_jfk_drop_off'] = distance(jfk[1], jfk[0], train_df.dropoff_latitude, train_df.dropoff_longitude)

        train_df['year'] = train_df.pickup_datetime.apply(lambda t: t.year)
        train_df['hour'] = train_df.pickup_datetime.apply(lambda t: t.hour)
        train_df['weekday'] = train_df.pickup_datetime.apply(lambda t: t.weekday())
        train_X = train_df[feature_classes]
        train_y = train_df["fare_amount"]

        train_X_values = train_X.values
        train_y_values = train_y.values
        for i in range(10):
            # sess.run([cost,train],feed_dict={xs:X_train, ys:y_train})
            for j in range(train_X.shape[0]):
                sess.run([cost, train], feed_dict={
                    xs: train_X_values[j, :].reshape(1, len(feature_classes)), ys:train_y_values[j]})
            epoch_cost = sess.run(cost, feed_dict={xs: train_X_values, ys: train_y_values})
            print('Epoch :', i, 'Cost :', epoch_cost)