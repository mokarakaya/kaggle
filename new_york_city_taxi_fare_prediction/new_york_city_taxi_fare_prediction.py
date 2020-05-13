import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.neural_network import MLPRegressor
import datetime

chunk_size = 100000


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

nyc = (-74.0063889, 40.7141667)
jfk = (-73.7822222222, 40.6441666667)

clf = MLPRegressor()

feature_classes = ["distance_miles", "year", "hour", 'weekday', 'passenger_count', 'month', 'day',
                   'distance_to_center', 'distance_to_center_drop_off']
counter = 0
print(datetime.datetime.now())
for i in range(0,2):
    for train_df in pd.read_csv('data/train_preprocessed.csv', chunksize=chunk_size):

        train_X = train_df[feature_classes]
        train_y = train_df["fare_amount"]

        X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.1, random_state=42)

        clf.partial_fit(X_train, y_train)
        val_preds = clf.predict(X_test)

        counter += 1
        print(i, counter, datetime.datetime.now(), np.sqrt(((val_preds - y_test) ** 2).mean()))

    test_df = pd.read_csv('data/test.csv', parse_dates=['pickup_datetime'])
    test_df['distance_miles'] = distance(test_df.pickup_latitude, test_df.pickup_longitude,
                                         test_df.dropoff_latitude, test_df.dropoff_longitude)
    test_df['distance_to_center'] = distance(nyc[1], nyc[0], test_df.pickup_latitude, test_df.pickup_longitude)
    test_df['distance_to_center_drop_off'] = distance(nyc[1], nyc[0], test_df.dropoff_latitude, test_df.dropoff_longitude)
    test_df['distance_to_jfk'] = distance(jfk[1], jfk[0], test_df.pickup_latitude, test_df.pickup_longitude)
    test_df['distance_to_jfk_drop_off'] = distance(jfk[1], jfk[0], test_df.dropoff_latitude, test_df.dropoff_longitude)

    test_df['year'] = test_df.pickup_datetime.apply(lambda t: t.year)
    test_df['hour'] = test_df.pickup_datetime.apply(lambda t: t.hour)
    test_df['weekday'] = test_df.pickup_datetime.apply(lambda t: t.weekday())
    test_df['month'] = test_df.pickup_datetime.apply(lambda t: t.month)
    test_df['day'] = test_df.pickup_datetime.apply(lambda t: t.day)

    test_X = test_df[feature_classes]
    test_df['fare_amount'] = clf.predict(test_X)
    test_df[['key', 'fare_amount']].to_csv('data/result.csv', index=False)
    print(datetime.datetime.now())
