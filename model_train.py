import os
import random
import numpy as np
import pandas as pd

from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from catboost import CatBoostRegressor, Pool

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.ensemble import AdaBoostRegressor


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    print("start")
    seed_everything(42)

    # df read
    train_df = pd.read_csv("data/TE_add_columns_train.csv")
    test_df = pd.read_csv("data/TE_add_columns_test.csv")

    test_x = test_df.drop(columns=["ID"]).copy()
    train_x = train_df[test_x.columns].copy()
    train_y = train_df["ECLO"].copy()

    train_x = train_x.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    test_x = test_x.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    # encoding
    categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)

    for i in categorical_features:
        te = TargetEncoder(cols=[i])
        train_x[i] = te.fit_transform(train_x[i], train_y)
        test_x[i] = te.transform(test_x[i])

    # model train
    train_preds = np.zeros(len(train_x))
    test_preds = np.zeros(len(test_x))

    # skf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
        dtrain = Pool(
            data=train_x.values[train_index], label=train_y.values[train_index]
        )
        dvalid = Pool(
            data=train_x.values[valid_index], label=train_y.values[valid_index]
        )
        bst = CatBoostRegressor(
            iterations=1000,
            objective="RMSE",
            learning_rate=0.01,
            l2_leaf_reg=6,
            depth=4,
            random_seed=42,
            subsample=0.7,
            bagging_temperature=0.23,
            od_type="Iter",
        )
        bst.fit(X=dtrain, eval_set=dvalid)
        test_preds += bst.predict(Pool(test_x.values)) / skf.n_splits

    # save submit csv
    sample_submission = pd.read_csv("data/sample_submission.csv")
    baseline_submission = sample_submission.copy()
    baseline_submission["ECLO"] = test_preds.astype(int)
    baseline_submission.to_csv("result/catboost_te_strfk0_mean.csv", index=False)
    print("end")

# seed_everything(42)

# train_df = pd.read_csv('data/modified_train.csv')
# test_df = pd.read_csv('data/modified_test.csv')

# test_x = test_df.drop(columns=['ID']).copy()
# train_x = train_df[test_x.columns].copy()
# train_y = train_df['ECLO'].copy()

# 레이블인코딩
# categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)

# for i in categorical_features:
#     le = LabelEncoder()
#     le = le.fit(train_x[i])
#     train_x[i] = le.transform(train_x[i])

#     for case in np.unique(test_x[i]):
#         if case not in le.classes_:
#             le.classes_ = np.append(le.classes_, case)
#     test_x[i] = le.transform(test_x[i])


# # xgb
# params = {
#     "objective": "reg:squaredlogerror",
#     "eval_metric": "rmsle",
#     "eta": 0.005,
#     "seed": 42,
#     "max_depth": 7,
#     "subsample": 0.8,
# }

# train_preds = np.zeros(len(train_x))
# test_preds = np.zeros(len(test_x))
# # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# skf = KFold(n_splits=5, shuffle=True, random_state=42)

# for i, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
#     dtrain = xgb.DMatrix(train_x.values[train_index], label=train_y.values[train_index])
#     dvalid = xgb.DMatrix(train_x.values[valid_index], label=train_y.values[valid_index])

#     bst = xgb.train(
#         params,
#         dtrain,
#         num_boost_round=1000,
#         evals=[(dtrain, "train"), (dvalid, "valid")],
#         verbose_eval=200,
#     )
#     train_preds[valid_index] = bst.predict(dvalid)
#     test_preds += bst.predict(xgb.DMatrix(test_x)) / skf.n_splits


# # lgb

# # def rmsle_lgbm(y_pred, data):

# #     y_true = np.array(data.get_label())
# #     score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

# #     return 'rmsle', score, False

# params = {
#     # 'objective': 'root_mean_squared_error',
#     "boosting_type": "gbdt",
#     "objective": "regression",
#     "learning_rate": 0.1,
#     "seed": 42,
#     "lambda_l1": 5,
#     "lambda_l2": 5,
#     "max_depth": 5,
#     "num_leaves": 20,
#     "force_col_wise": True,
#     "nthread": 24,
#     "bagging_fraction": 0.6,
#     "bagging_frequency": 10,
#     "feature_fraction": 0.5,
#     "metric": "rmse",
# }

# train_preds = np.zeros(len(train_x))
# test_preds = np.zeros(len(test_x))
# # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# skf = KFold(n_splits=5, shuffle=True, random_state=42)
# for i, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
#     dtrain = lgb.Dataset(train_x.values[train_index], label=train_y.values[train_index])
#     dvalid = lgb.Dataset(train_x.values[valid_index], label=train_y.values[valid_index])

#     bst = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid])
#     # train_preds[valid_index] = bst.predict(dvalid)
#     # test_preds += bst.predict(lgb.Dataset(test_x)) / skf.n_splits
#     test_preds += bst.predict(test_x.values) / skf.n_splits


# cb
# custom metric 존재

# params = {
#     "iterations ": 1000,
#     "objective": "RMSE",
#     "learning_rate": 0.01,
#     "l2_leaf_reg": 3,
#     "depth": 4,
#     "random_seed": 42,
#     "subsample": 0.6,
#     "bagging_temperature": 0.2,
#     "od_type": "Iter",
# }

# train_preds = np.zeros(len(train_x))
# test_preds = np.zeros(len(test_x))

# skf = KFold(n_splits=5, shuffle=True, random_state=42)
# # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# for i, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
#     dtrain = Pool(data=train_x.values[train_index], label=train_y.values[train_index])
#     dvalid = Pool(data=train_x.values[valid_index], label=train_y.values[valid_index])
#     bst = CatBoostRegressor(
#         iterations=1000,
#         objective="RMSE",
#         learning_rate=0.01,
#         l2_leaf_reg=3,
#         depth=4,
#         random_seed=42,
#         subsample=0.7,
#         bagging_temperature=0.23,
#         od_type="Iter",
#     )
#     bst.fit(X=dtrain, eval_set=dvalid)
#     test_preds += bst.predict(Pool(test_x.values)) / skf.n_splits


# Adaboost
# custom metric

# train_preds = np.zeros(len(train_x))
# test_preds = np.zeros(len(test_x))

# skf = KFold(n_splits=5, shuffle=True, random_state=42)

# for i, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
#     X_train, X_val = train_x.values[train_index], train_y.values[train_index]
#     y_train, y_val = train_x.values[valid_index], train_y.values[valid_index]
#     bst = AdaBoostRegressor(n_estimators=500, learning_rate=0.01,
#                             random_state=42, loss='square')
#     print(X_train.shape, y_train.shape)
#     print(X_val.shape, y_val.shape)
#     bst.fit(X_train, X_val)
#     loss = bst.score(y_train, y_val)
#     print(f'fold-{i}, l2loss: {loss}')
#     test_preds += bst.predict(test_x.values) / skf.n_splits

# sample_submission = pd.read_csv("data/sample_submission.csv")
# baseline_submission = sample_submission.copy()
# baseline_submission["ECLO"] = test_preds.astype(int)
# baseline_submission.to_csv("result/cbcbcb.csv", index=False)
