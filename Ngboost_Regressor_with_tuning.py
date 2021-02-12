#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_squared_error
from hyperopt import hp, tpe, space_eval
from hyperopt.pyll.base import scope
from hyperopt.fmin import fmin
from hyperopt import STATUS_OK, Trials
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import logging

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# NGBoost regressor使用tuning结果
logging.basicConfig(filename="ngboost_output.log",
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

boston = load_boston()
data = boston.data
target = boston.target

# 2次数据划分，这样可以分成3份数据  test  train  validation
X_intermediate, X_test, y_intermediate, y_test = train_test_split(
    data, target, shuffle=True, test_size=0.2, random_state=1)

# train/validation split (gives us train and validation sets)
X_train, X_validation, y_train, y_validation = train_test_split(X_intermediate,
                                                                y_intermediate,
                                                                shuffle=False,
                                                                test_size=0.25,
                                                                random_state=1)

# delete intermediate variables
del X_intermediate, y_intermediate

# print proportions
# 显示数据集的分配比例
print('train: {}% | validation: {}% | test {}%'.format(
    round(len(y_train) / len(target), 2),
    round(len(y_validation) / len(target), 2),
    round(len(y_test) / len(target), 2)))

# 搜索空间设定
b1 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=2)
b2 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)
b3 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4)

space = {
    'learning_rate': hp.uniform('learning_rate', .05, 1),
    'minibatch_frac': hp.choice('minibatch_frac', [1.0, 0.5]),
    'Base': hp.choice('Base', [b1, b2, b3])
}

# n_estimators表示一套参数下，有多少个评估器，简单说就是迭代多少次
default_params = {"n_estimators": 20, "verbose_eval": 1, "random_state": 1}


def objective(params):

    params.update(default_params)

    print("current params:", params)
    ngb = NGBRegressor(**params).fit(
        X_train,
        y_train,
        X_val=X_validation,
        Y_val=y_validation,
        #  假定n_estimators迭代器有100个设定了早期停止后也许不到100次迭代就完成了训练停止了
        early_stopping_rounds=2)
    loss = ngb.evals_result['val']['LOGSCORE'][ngb.best_val_loss_itr]
    logger.info("current params:{}".format(params))
    results = {'loss': loss, 'status': STATUS_OK}

    return results


TRIALS = Trials()
logger.info("Start parameter optimization...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        # max_evals是设定多少套参数组合，组合数越大准确度可能更高但是训练的时间越长
        max_evals=200,
        trials=TRIALS)

best_params = space_eval(space, best)
logger.info("best params:{}".format(best_params))
logger.info("...done")

ngb_new = NGBRegressor(**best_params).fit(
    X_train,
    y_train,
    X_val=X_validation,
    Y_val=y_validation,
    #  假定n_estimators迭代器有100个设定了早期停止后也许不到100次迭代就完成了训练停止了
    early_stopping_rounds=2)

y_pred = ngb_new.predict(X_test)
test_MSE = mean_squared_error(y_pred, y_test)
print('Test MSE_ngb_new', test_MSE)


def lr_loss_plot(ft, trials):

    print("Loss plot for parameter {}".format(ft))

    _loss = list()
    _ft = list()

    for t in trials.trials:
        try:
            if len(t['misc']['vals'][ft]) > 0:
                _ft.append(t['misc']['vals'][ft][0])
                _loss.append(t['result']['loss'])
        except:
            pass

    data_lr = pd.DataFrame([_loss, _ft]).T
    data_lr.columns = ['loss', ft]

    sns.lineplot(y='loss', x=ft, data=data_lr)
    plt.show()


def count_loss_plot(ft, trials):

    print("learning_rate plot for parameter {}".format(ft))

    _loss = list()
    _ft = list()

    for t in trials.trials:
        _ft.append(t['tid'])
        _loss.append(t['result']['loss'])

    data_tid = pd.DataFrame([_loss, _ft]).T
    data_tid.columns = ['loss', ft]

    sns.lineplot(y='loss', x=ft, data=data_tid)
    plt.show()


lr_loss_plot('learning_rate', TRIALS)
count_loss_plot('count', TRIALS)

# if __name__ == "__main__":
