import functools
import xgboost

import pandas as pd
import numpy as np
from scipy.stats import skew, norm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import model_selection

# Data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')


# Data Preprocessing
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
df_train = df_train.drop(df_train.loc[df_train['Embarked'].isnull()].index)
df_train = df_train.drop("PassengerId", 1)

df_train_X = df_train.drop("Survived", 1)
df_train_Y = df_train["Survived"]


def substring(s_list, s):
    if not type(s) == float:
        for title in s_list:
            if title in s:
                return title
    return "Unknown"


# replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

def get_person(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    elif age > 55:
        return 'elderly'
    else:
        return sex

def family_size(num):
    if num == 1:
        return "single"
    elif num > 4:
        return "large"
    else:
        return "small"

def transform_df(df):

    df["Title"] = df["Name"].map(
        functools.partial(substring,
                          ['Mrs.', 'Mr.', 'Miss.']
                          ))
    # df['Title'] = df.apply(replace_titles, axis=1)
    df["Family"] = df["SibSp"] + df["Parch"] + 1
    # df["Family_Size"] = df["Family"].map(family_size)
    df["Has_Cabin"] = df["Cabin"].isnull()
    # df['Person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
    # df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    # df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    df = df.drop(["Name", "Ticket"], 1)
    df = df.drop((percent[percent > 0.5]).index, 1)
    return df

df_train_X = transform_df(df_train_X)
# print(df_train_X)
categorical_features = [(idx, f) for idx, f in enumerate(df_train_X.columns) if df_train_X.dtypes[f] == 'object']
numerical_features = [(idx, f) for idx, f in enumerate(df_train_X.columns) if df_train_X.dtypes[f] != 'object']

category_mappings = {}
log_features = []
for idx, f in categorical_features:
    category_mappings[f] = {label: i for i, label in enumerate(np.unique(df_train_X[f]))}
    df_train_X[f] = df_train_X[f].map(category_mappings[f])
for idx, f in numerical_features:
    if skew(df_train_X[f]) > 0.75:
        log_features.append(f)
        df_train_X[f] = np.log1p(df_train_X[f])

df_train_X = df_train_X.values
df_train_Y = df_train_Y.values
imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
df_train_X = imp.fit_transform(df_train_X)

enc = preprocessing.OneHotEncoder(categorical_features=[idx for idx, _ in categorical_features])
df_train_X = enc.fit_transform(df_train_X)

scaler = preprocessing.StandardScaler(with_mean=False)
df_train_X = scaler.fit_transform(df_train_X)

# Grid Search
# reg = linear_model.LogisticRegression(max_iter=50000)
# clf = model_selection.GridSearchCV(reg, {'C': [0.1, 0.5, 1.0, 5, 10]})
# clf.fit(df_train_X, df_train_Y)
# print(clf.best_params_)
# print(clf.best_score_)

# reg = svm.SVC()
# clf = model_selection.GridSearchCV(reg, {'C': [0.1, 0.5, 1.0, 5, 10]})
# clf.fit(df_train_X, df_train_Y)
# print(clf.best_params_)
# print(clf.best_score_)

# reg = ensemble.RandomForestClassifier()
# clf = model_selection.GridSearchCV(reg, {'n_estimators': [5, 10, 20, 50, 100, 200]})
# clf.fit(df_train_X, df_train_Y)
# print(clf.best_params_)
# print(clf.best_score_)

reg = xgboost.XGBClassifier(
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1)
clf = model_selection.GridSearchCV(reg, {'gamma': [0.3, 0.5, 0.7],
                                         'n_estimators': [50, 100, 500]})
clf.fit(df_train_X, df_train_Y)
print(clf.best_params_)
print(clf.best_score_)


df_test_ID = df_test["PassengerId"][:, np.newaxis]
df_test = df_test.drop("PassengerId", 1)
df_test = transform_df(df_test)
for idx, f in categorical_features:
    df_test[f] = df_test[f].map(category_mappings[f])
for f in log_features:
    df_test[f] = np.log1p(df_test[f])
df_test_X = df_test.values
df_test_X = imp.transform(df_test_X)
df_test_X = enc.transform(df_test_X)
df_test_X = scaler.transform(df_test_X)
df_test_Y = clf.predict(df_test_X)
df_result = pd.DataFrame(np.concatenate((df_test_ID, [[item] for item in df_test_Y]), axis=1), columns=['PassengerId', 'Survived'])
df_result["PassengerId"] = df_result["PassengerId"].astype(int)
df_result.to_csv("data/submission_xgboost.csv", index=False)

