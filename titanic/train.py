import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model as lm
import numpy as np
from titanic_util import fill_na, encode_label


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

fill_na(train_df, test_df, 'Fare', train_df["Fare"].median())
fill_na(train_df, test_df, 'Age', train_df["Age"].median())
fill_na(train_df, test_df, 'Embarked', 'S')

encode_label(train_df, test_df, 'Sex', 'SexE')
encode_label(train_df, test_df, 'Embarked', 'EmbarkedE')

train_df['Family_Size'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['Family_Size'] = test_df['SibSp'] + test_df['Parch'] + 1

train_df['Is_Alone'] = 0
test_df['Is_Alone'] = 0
train_df.loc[train_df['Family_Size'] == 1, 'Is_Alone'] = 1
test_df.loc[test_df['Family_Size'] == 1, 'Is_Alone'] = 1


feature_classes = ["Pclass", "SibSp", "Parch", "Fare", 'SexE', 'Age', 'EmbarkedE', 'Family_Size', 'Is_Alone']
train_X = train_df[feature_classes]
train_y = train_df["Survived"]
test_X = test_df[feature_classes]


X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)
# clf = MLPClassifier(solver='adam',
#                     alpha=1e-5,
#                     hidden_layer_sizes=(100, 8),
#                     random_state=1,
#                     activation='tanh',
#                     learning_rate='adaptive')
# clf = BaggingClassifier()
clf = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=100)
clf = BaggingClassifier(base_estimator=clf, verbose=0, n_jobs=-1, random_state =1)

# clf = lm.LogisticRegression()


clf.fit(X_train, y_train)
val_preds = clf.predict(X_test)

print(accuracy_score(y_test, val_preds))

test_df['Survived'] = clf.predict(test_X)
test_df[['PassengerId', 'Survived']].to_csv('data/result.csv', index=False)


