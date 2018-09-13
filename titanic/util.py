from sklearn import preprocessing


def encode_label(train_df, test_df, label, encoded_label):
    le = preprocessing.LabelEncoder()
    le.fit(train_df[label])
    train_df[encoded_label] = le.transform(train_df[label])
    test_df[encoded_label] = le.transform(test_df[label])


def fill_na(train_df, test_df, label, value):
    train_df[label].fillna(value, inplace=True)
    test_df[label].fillna(value, inplace=True)
