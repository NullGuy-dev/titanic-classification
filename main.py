import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

test_fn = "test.csv"
train_fn = "train.csv"
test_labels_fn = "gender_submission.csv"

train_data = pd.read_csv(train_fn)
test_data = pd.read_csv(test_fn)
test_labels_data = pd.read_csv(test_labels_fn)

def norm_age(age):
    return 0 if pd.isnull(age) else age

def sex2num(sex):
    return 1 if sex == "male" else 0

def preproc_data(data, is_training=False):
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    data['Age'] = data['Age'].apply(norm_age)
    data['Sex'] = data['Sex'].apply(sex2num)
    
    scaler = StandardScaler()
    data['Fare'] = scaler.fit_transform(data['Fare'].values.reshape(-1, 1))
    
    if is_training:
        y = data.pop('Survived')
        return data, y
    return data

X_train, y_train = preproc_data(train_data, is_training=True)
X_test = preproc_data(test_data)
y_test = test_labels_data['Survived']

model = RandomForestClassifier(max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score:.2f}")