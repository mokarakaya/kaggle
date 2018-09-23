import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.neural_network import MLPRegressor
import datetime
import os

chunk_size = 500000

nyc = (-74.0063889, 40.7141667)
jfk = (-73.7822222222, 40.6441666667)

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


feature_classes = ['key', "distance_miles", "year", "hour", 'weekday', 'passenger_count', 'month', 'day',
                   'distance_to_center', 'distance_to_center_drop_off',
                   'distance_to_jfk', 'distance_to_jfk_drop_off', 'fare_amount']

print(datetime.datetime.now())
counter = 0
for train_df in pd.read_csv('data/train.csv', chunksize=chunk_size, parse_dates=['pickup_datetime']):

    train_df.dropna(how='any', inplace=True)

    train_df['distance_miles'] = distance(train_df.pickup_latitude, train_df.pickup_longitude,
                                          train_df.dropoff_latitude, train_df.dropoff_longitude).round(3)
    train_df['distance_to_center'] = distance(nyc[1], nyc[0], train_df.pickup_latitude, train_df.pickup_longitude)
    train_df['distance_to_center_drop_off'] = distance(nyc[1], nyc[0], train_df.dropoff_latitude, train_df.dropoff_longitude)

    train_df['distance_to_jfk'] = distance(jfk[1], jfk[0], train_df.pickup_latitude, train_df.pickup_longitude)
    train_df['distance_to_jfk_drop_off'] = distance(jfk[1], jfk[0], train_df.dropoff_latitude, train_df.dropoff_longitude)

    train_df['year'] = train_df.pickup_datetime.apply(lambda t: t.year)
    train_df['hour'] = train_df.pickup_datetime.apply(lambda t: t.hour)
    train_df['weekday'] = train_df.pickup_datetime.apply(lambda t: t.weekday())
    train_df['month'] = train_df.pickup_datetime.apply(lambda t: t.month)
    train_df['day'] = train_df.pickup_datetime.apply(lambda t: t.day)

    train_X = train_df[feature_classes]
    if not os.path.isfile('data/train_preprocessed.csv'):
        train_X.to_csv('data/train_preprocessed.csv', index=False)
    else:  # else it exists so append without writing the header
        train_X.to_csv('data/train_preprocessed.csv', index=False, mode='a', header=False)
    counter += 1
    print(counter, datetime.datetime.now())


