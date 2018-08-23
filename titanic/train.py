import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import linear_model as lm
import numpy as np

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
le = preprocessing.LabelEncoder()

le.fit(train_df['Sex'])
train_df['SexL'] = le.transform(train_df['Sex'])
test_df['SexL'] = le.transform(test_df['Sex'])

train_X = train_df[["Pclass", "SibSp", "Parch", "Fare", 'SexL']]
train_y = train_df["Survived"]
test_X = test_df[["Pclass", "SibSp", "Parch", "Fare", 'SexL']]

test_X["Fare"].fillna(test_X["Fare"].median(), inplace=True)

dev_X, val_X, dev_y, val_y = train_test_split(train_X, train_y, test_size=0.33, random_state=42)
clf = MLPClassifier(solver='adam',
                    alpha=1e-5,
                    hidden_layer_sizes=(100, 8),
                    random_state=1,
                    activation='tanh',
                    learning_rate='adaptive')
# clf = RandomForestClassifier(max_depth= 2, random_state=0)
# clf = lm.LogisticRegression()


clf.fit(dev_X, dev_y)
val_preds = clf.predict(val_X)

print(accuracy_score(val_y, val_preds))

test_df['Survived'] = clf.predict(test_X)
test_df[['PassengerId', 'Survived']].to_csv('data/result.csv', index=False)
