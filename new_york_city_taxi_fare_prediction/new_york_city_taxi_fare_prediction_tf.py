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

nyc = (-74.0063889, 40.7141667)
jfk = (-73.7822222222, 40.6441666667)

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


feature_classes = ["distance_miles",
                   "hour", 'weekday', 'passenger_count']



def neural_net_model(X_data,input_dim):
    hidden_layer_nodes = 32
    A1 = tf.Variable(tf.random_normal(mean=0.5, stddev=0.5, shape=[input_dim, hidden_layer_nodes]))  # inputs -> hidden nodes
    b1 = tf.Variable(tf.random_normal(mean=0.5, stddev=0.5, shape=[hidden_layer_nodes]))  # one biases for each hidden node
    A2 = tf.Variable(tf.random_normal(mean=0.5, stddev=0.5, shape=[hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
    b2 = tf.Variable(tf.random_normal(mean=0.5, stddev=0.5, shape=[1]))  # 1 bias for the output

    # Declare model operations
    hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, A1), b1))
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

    return final_output

print(datetime.datetime.now())
xs = tf.placeholder("float")
ys = tf.placeholder("float")

output = neural_net_model(xs, len(feature_classes))

cost = tf.reduce_mean(tf.square(output - ys))
# train = tf.train.AdamOptimizer(0.05).minimize(cost)
train = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

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
        epoch_costs = []
        for i in range(10):
            # sess.run([cost,train],feed_dict={xs:X_train, ys:y_train})
            for j in range(train_X.shape[0]):
                sess.run([train, cost], feed_dict={
                    xs: train_X_values[j, :].reshape(1, len(feature_classes)), ys:train_y_values[j]})
                epoch_cost = sess.run(cost, feed_dict={
                    xs: train_X_values[j, :].reshape(1, len(feature_classes)), ys: train_y_values[j]})
                epoch_costs.append(epoch_cost)
            print('Epoch :', i, 'Cost :', np.average(epoch_costs))