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
import numpy as np

# 显示中文方法
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

ngb = NGBRegressor().fit(
    X_train,
    y_train,
    X_val=X_validation,
    Y_val=y_validation,
    #  假定n_estimators迭代器有100个设定了早期停止后也许不到100次迭代就完成了训练停止了
    early_stopping_rounds=2)

y_pred = ngb.predict(X_test)
print("y_pred=", y_pred)
print("y_test=", y_test)
test_MSE = mean_squared_error(y_pred, y_test)
print('Test MSE_ngb', test_MSE)

logger.info("...done")

plt.figure(figsize=(8, 6))
plt.scatter(x=y_pred, y=y_test, s=20)
# 创建一条斜线，然后让2个值作为x和y输出，如果完全相同那就会和斜线重合，越靠近斜线说明拟合效果越好
plt.plot([8, 50], [8, 50], color="gray", ls="--")
plt.xlabel("NGBoost predict values")
plt.ylabel("Actual values")
plt.title(
    "Compare NGBoost probability predict and actual values(越靠近斜线说明拟合效果越好)")
plt.show()

# if __name__ == "__main__":
