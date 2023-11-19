import numpy as np
import pandas as pd

# catboost custom metric 적용하기
class RMSLEObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        """
        Computes first and second derivative of the loss function 
        with respect to the predicted value for each object.

        Parameters
        ----------
        approxes : indexed container of floats
            Current predictions for each object.

        targets : indexed container of floats
            Target values you provided with the dataset.

        weight : float, optional (default=None)
            Instance weight.

        Returns
        -------
            der1 : list-like object of float
            der2 : list-like object of float

        """
        pass
    


def RMSLE_CB(y, t):
    t = t.get_label()
    # print(y.shape, t.shape)
    log_y = np.log1p(y)
    log_t = np.log1p(t)
    loss = np.sqrt(np.mean((log_y - log_t)**2))
    return 'rmsle', loss, False


params = {
    "iterations ": 1000,
    "objective": "RMSE",
    "learning_rate": 0.01,
    "l2_leaf_reg": 3,
    "depth": 4,
    "random_seed": 42,
    "subsample": 0.6,
    "bagging_temperature": 0.2,
    "od_type": "Iter",
}

train_preds = np.zeros(len(train_x))
test_preds = np.zeros(len(test_x))

skf = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
    dtrain = Pool(data=train_x.values[train_index], label=train_y.values[train_index])
    dvalid = Pool(data=train_x.values[valid_index], label=train_y.values[valid_index])
    bst = CatBoostRegressor(
        iterations=1000,
        loss_function='RMSLEObjective'
        learning_rate=0.01,
        l2_leaf_reg=4,
        depth=4,
        random_seed=42,
        subsample=0.6,
        bagging_temperature=0.23,
        od_type="Iter",
    )
    bst.fit(X=dtrain, eval_set=dvalid)
    test_preds += bst.predict(Pool(test_x.values)) / skf.n_splits


# submit
sample_submission = pd.read_csv("data/sample_submission.csv")
baseline_submission = sample_submission.copy()
baseline_submission["ECLO"] = test_preds.astype(int)
baseline_submission.to_csv("result/baseline_submit.csv", index=False)
