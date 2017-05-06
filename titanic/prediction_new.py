import functools
import xgboost
import re

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


# Data Preprocessing, Fill NA
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
df_train["Embarked"] = df_train["Embarked"].fillna('C')
df_train = df_train.drop("PassengerId", 1)


def fill_missing_fare(df):
    median_fare = df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df

df_test = fill_missing_fare(df_test)

df_train_X = df_train.drop("Survived", 1)
df_train_Y = df_train["Survived"]


# Feature Engineering
def substring(s_list, s):
    if not type(s) == float:
        for title in s_list:
            if title in s:
                return title
    return "Rare"


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
    df["Family"] = df["SibSp"] + df["Parch"] + 1
    # df["Is_Alone"] = df["Family"].map(lambda x: x < 2)
    # df["Family_Size"] = df["Family"].map(family_size)

    df["Has_Cabin"] = df["Cabin"].isnull()
    # df["NameLength"] = df["Name"].apply(lambda x: len(x))
    # df['Person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
    # df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    # age_avg = df['Age'].mean()
    # age_std = df['Age'].std()
    # age_null_count = df['Age'].isnull().sum()
    # age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # df['Age'][np.isnan(df['Age'])] = age_null_random_list
    # df['Age'] = df['Age'].astype(int)

    df = df.drop(["Name", "Ticket", "Cabin"], 1)
    return df

df_train_X = transform_df(df_train_X)
categorical_features = [(idx, f) for idx, f in enumerate(df_train_X.columns)
                        if df_train_X.dtypes[f].name in ['object', 'category']]
numerical_features = [(idx, f) for idx, f in enumerate(df_train_X.columns)
                      if df_train_X.dtypes[f].name not in ['object', 'category']]
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
# clf_lr = model_selection.GridSearchCV(reg, {'C': [0.1, 0.5, 1.0, 5]})
# clf_lr.fit(df_train_X, df_train_Y)
# print(clf_lr.best_params_)
# print(clf_lr.best_score_)
#
reg = svm.SVC()
clf_svc = model_selection.GridSearchCV(reg, {'C': [0.1,0.3, 0.5, 0.7,1,5]}, cv=5)
clf_svc.fit(df_train_X, df_train_Y)
print(clf_svc.best_params_)
print(clf_svc.best_score_)
#
# reg = ensemble.RandomForestClassifier(
#     n_jobs=-1,
# )
# clf_rf = model_selection.GridSearchCV(reg, {'n_estimators': [50, 100, 200, 500]})
# clf_rf.fit(df_train_X, df_train_Y)
# print(clf_rf.best_params_)
# print(clf_rf.best_score_)

# reg = ensemble.GradientBoostingClassifier(
#     max_depth=5,
#     min_samples_leaf=3,
#     verbose=3
# )
# clf_gb = model_selection.GridSearchCV(reg, {'n_estimators': [500]})
# clf_gb.fit(df_train_X.toarray(), df_train_Y)
# print(clf_gb.best_params_)
# print(clf_gb.best_score_)

# reg = ensemble.AdaBoostClassifier(
#     learning_rate=0.95
# )
# clf_ada = model_selection.GridSearchCV(reg, {'n_estimators': [5, 10, 20, 50]})
# clf_ada.fit(df_train_X, df_train_Y)
# print(clf_ada.best_params_)
# print(clf_ada.best_score_)


# df_train_lr_X = np.array(clf_lr.predict(df_train_X)).reshape(-1, 1)
# df_train_svc_X = np.array(clf_svc.predict(df_train_X)).reshape(-1, 1)
# df_train_rf_X = np.array(clf_rf.predict(df_train_X)).reshape(-1, 1)
# df_train_gb_X = np.array(clf_gb.predict(df_train_X.toarray())).reshape(-1, 1)
# df_train_ada_X = np.array(clf_ada.predict(df_train_X)).reshape(-1, 1)
#
# df_train_all_X = np.concatenate((df_train_lr_X, df_train_svc_X, df_train_rf_X, df_train_gb_X, df_train_ada_X), axis=1)
#
# reg = xgboost.XGBClassifier(
#     max_depth=4,
#     min_child_weight=2,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='binary:logistic',
#     nthread=-1,
#     scale_pos_weight=1)
# clf = model_selection.GridSearchCV(reg, {'gamma': [0.5, 1],
#                                          'n_estimators': [100, 1000, 10000]})
# clf.fit(df_train_all_X, df_train_Y)
# print(clf.best_params_)
# print(clf.best_score_)


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

# df_test_lr_X = np.array(clf_lr.predict(df_test_X)).reshape(-1, 1)
# df_test_svc_X = np.array(clf_svc.predict(df_test_X)).reshape(-1, 1)
# df_test_rf_X = np.array(clf_rf.predict(df_test_X)).reshape(-1, 1)
# df_test_gb_X = np.array(clf_gb.predict(df_test_X.toarray())).reshape(-1, 1)
# df_test_ada_X = np.array(clf_ada.predict(df_test_X)).reshape(-1, 1)
#
# df_test_all_X = np.concatenate((df_test_lr_X, df_test_svc_X, df_test_rf_X, df_test_gb_X, df_test_ada_X), axis=1)

df_test_Y = clf_svc.predict(df_test_X.toarray())
df_result = pd.DataFrame(np.concatenate((df_test_ID, [[item] for item in df_test_Y]), axis=1), columns=['PassengerId', 'Survived'])
df_result["PassengerId"] = df_result["PassengerId"].astype(int)
df_result.to_csv("data/submission_SVM.csv", index=False)
