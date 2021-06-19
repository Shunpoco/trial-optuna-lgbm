import numpy as np
import optuna

import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics

from sklearn.model_selection import train_test_split

def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_X, valid_X, train_y, valid_y = train_test_split(data, target, test_size=0.25)

    dtrain = lgb.Dataset(train_X, label=train_y)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_X)
    pred_labels = np.rint(preds)

    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print(f'Number of finished trials: {len(study.trials)}')

    print('Best trial')
    trial = study.best_trial

    print(f'    Value: {trial.value}')

    print(f'    Params: ')
    for key, value in trial.params.items():
        print(f'        {key}: {value}')
