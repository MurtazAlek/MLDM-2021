import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt



#TASK_1
data = pd.read_csv(r'C:\Users\Aleksandra\OneDrive\Desktop\MachineLearning\MLDM_2020\train.csv', index_col='PassengerId')
pvt = data.pivot_table(index=['Pclass'], values=['Survived'],aggfunc=np.mean)
grp_by=(data[['Pclass','Survived']].set_index('Pclass')).groupby(level='Pclass').mean()

#TASK_3
def feature_selection_and_preprocessing(data):
    features = data.copy()
    features['Age'] = features['Age'].interpolate(method='linear', inplace=False)
    features['Age'] = features['Age'].fillna(features['Age'].mean())
    features['Sex'] = features['Sex'].replace({"male": int(0), "female": int(1)}, inplace=False)
    features["Fare"] /= features["Fare"].mean()
    dummy = pd.get_dummies(features['Embarked'])
    features = pd.merge(features, dummy, left_index=True, right_index=True)
    features.rename(columns={'C': 'Embarked_C', 'S': 'Embarked_S', 'Q': 'Embarked_Q'}, inplace=False)
    label = ['Name', 'Ticket', 'Cabin', 'Embarked']
    features = features.drop(labels=label, axis=1)
    return features

# Validation code (do not touch)
data = pd.read_csv(r'C:\Users\Aleksandra\OneDrive\Desktop\MachineLearning\MLDM_2020\train.csv', index_col='PassengerId')
data_train = data.iloc[:-100]
data_test = data.iloc[-100:]

model = KNeighborsClassifier(
    weights='uniform',
    n_neighbors=8
)

model.fit(
    feature_selection_and_preprocessing(
        data_train.drop('Survived', axis=1)
    ),
    data_train['Survived']
)

test_predictions = model.predict(
    feature_selection_and_preprocessing(
        data_test.drop('Survived', axis=1)
    )
)
print("Test accuracy:", accuracy_score(
    data_test['Survived'],
    test_predictions
))




#TASK_4

X_train,X_test,Y_train,Y_test = train_test_split(feature_selection_and_preprocessing(data).drop('Survived', axis=1),
                                                 feature_selection_and_preprocessing(data)['Survived'],test_size=0.11, random_state=16)

accuracy_dict={}

for number in np.random.randint(10,1000,500):
    X_train, X_test, Y_train, Y_test = train_test_split(
        feature_selection_and_preprocessing(data).drop('Survived', axis=1),
        feature_selection_and_preprocessing(data)['Survived'], test_size=0.11, random_state=number)
    model.fit(X_train,Y_train)
    test_predictions=model.predict(X_test)
    accuracy=accuracy_score(Y_test,test_predictions)
    accuracy_dict[number]=accuracy
print(accuracy_dict)

plt.hist(accuracy_dict.values())
plt.xlabel("accuracy")
plt.grid()
plt.title('histogram of accuracy')
plt.show()