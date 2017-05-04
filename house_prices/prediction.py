import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection

# Data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')


# Data Preprocessing
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
df_train = df_train.drop("Id", 1)

df_train_X = df_train.drop("SalePrice", 1)
df_train_Y = np.log(df_train["SalePrice"])


def transform_df(df):
    df = df.drop((percent[percent > 0.0001]).index, 1)
    df['GrLivArea'] = np.log(df['GrLivArea'])
    df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
    df['HasBsmt'] = 0
    df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
    # Encode some categorical features as ordered numbers when there is information in the order
    df = df.replace({"ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                                    "Min2": 6, "Min1": 7, "Typ": 8},
                     "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                     "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
                     "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
                     "PavedDrive": {"N": 0, "P": 1, "Y": 2},
                     "Street": {"Grvl": 1, "Pave": 2},
                     "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}
                     })
    # Create new features
    # 1* Simplifications of existing features
    df["SimplOverallQual"] = df.OverallQual.replace({1: 1, 2: 1, 3: 1,  # bad
                                                     4: 2, 5: 2, 6: 2,  # average
                                                     7: 3, 8: 3, 9: 3, 10: 3  # good
                                                     })
    df["SimplOverallCond"] = df.OverallCond.replace({1: 1, 2: 1, 3: 1,  # bad
                                                     4: 2, 5: 2, 6: 2,  # average
                                                     7: 3, 8: 3, 9: 3, 10: 3  # good
                                                     })
    df["SimplFunctional"] = df.Functional.replace({1: 1, 2: 1,  # bad
                                                   3: 2, 4: 2,  # major
                                                   5: 3, 6: 3, 7: 3,  # minor
                                                   8: 4  # typical
                                                   })
    df["SimplKitchenQual"] = df.KitchenQual.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
    df["SimplHeatingQC"] = df.HeatingQC.replace({1: 1,  # bad
                                                 2: 1, 3: 1,  # average
                                                 4: 2, 5: 2  # good
                                                 })
    # 2* Combinations of existing features
    # Overall quality of the house
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df["TotalBath"] = df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"]) + \
                      df["FullBath"] + (0.5 * df["HalfBath"])
    df["AllSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["AllFlrsSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
    # 3* Polynomials on the top 10 existing features
    df["OverallQual-s2"] = df["OverallQual"] ** 2
    df["OverallQual-s3"] = df["OverallQual"] ** 3
    df["OverallQual-Sq"] = np.sqrt(df["OverallQual"])
    df["AllSF-2"] = df["AllSF"] ** 2
    df["AllSF-3"] = df["AllSF"] ** 3
    df["AllSF-Sq"] = np.sqrt(df["AllSF"])
    df["AllFlrsSF-2"] = df["AllFlrsSF"] ** 2
    df["AllFlrsSF-3"] = df["AllFlrsSF"] ** 3
    df["AllFlrsSF-Sq"] = np.sqrt(df["AllFlrsSF"])
    df["GrLivArea-2"] = df["GrLivArea"] ** 2
    df["GrLivArea-3"] = df["GrLivArea"] ** 3
    df["GrLivArea-Sq"] = np.sqrt(df["GrLivArea"])
    df["SimplOverallQual-s2"] = df["SimplOverallQual"] ** 2
    df["SimplOverallQual-s3"] = df["SimplOverallQual"] ** 3
    df["SimplOverallQual-Sq"] = np.sqrt(df["SimplOverallQual"])
    return df

df_train_X = transform_df(df_train_X)
categorical_features = [(idx, f) for idx, f in enumerate(df_train_X.columns) if df_train_X.dtypes[f] == 'object']
numerical_features = [(idx, f) for idx, f in enumerate(df_train_X.columns) if df_train_X.dtypes[f] != 'object']

category_mappings = {}
log_features = []
for idx, f in categorical_features:
    category_mappings[f] = {label: i for i, label in enumerate(np.unique(df_train_X[f]))}
    df_train_X[f] = df_train_X[f].map(category_mappings[f])
for idx, f in numerical_features:
    if skew(df_train_X[f]) > 0.8:
        log_features.append(f)
        df_train_X[f] = np.log1p(df_train_X[f])

df_train_X = df_train_X.values
df_train_Y = df_train_Y.values

imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
df_train_X = imp.fit_transform(df_train_X)

enc = preprocessing.OneHotEncoder(categorical_features=[idx for idx, _ in categorical_features])
df_train_X = enc.fit_transform(df_train_X)

# scaler = preprocessing.StandardScaler(with_mean=False)
# df_train_X = scaler.fit_transform(df_train_X)

# Grid Search
reg = linear_model.Lasso(max_iter=50000)
clf = model_selection.GridSearchCV(reg, {'alpha': [0.0003, 0.0005, 0.0007]})
clf.fit(df_train_X, df_train_Y)
print(clf.best_params_)
print(clf.best_score_)


df_test_ID = df_test["Id"][:, np.newaxis]
df_test = df_test.drop("Id", 1)
df_test = transform_df(df_test)
for idx, f in categorical_features:
    df_test[f] = df_test[f].map(category_mappings[f])
for f in log_features:
    df_test[f] = np.log1p(df_test[f])
df_test_X = df_test.values
df_test_X = imp.transform(df_test_X)
df_test_X = enc.transform(df_test_X)
# df_test_X = scaler.transform(df_test_X)
df_test_Y = np.exp(clf.predict(df_test_X))
df_result = pd.DataFrame(np.concatenate((df_test_ID, [[item] for item in df_test_Y]), axis=1), columns=['Id', 'SalePrice'])
df_result["Id"] = df_result["Id"].astype(int)
df_result.to_csv("data/submission_lasso.csv", index=False)

