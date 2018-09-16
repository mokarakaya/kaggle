def fill_na(train_df, label, value):
    train_df[label].fillna(value, inplace=True)
