from ngboost import NGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime

# 显示中文方法
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# ***************************************************************************************
# 全体程序例子参考了下面链接的内容
# https://github.com/stanfordmlgroup/ngboost/blob/master/examples/tuning/hyperopt.ipynb

# ***************************************************************************************


def lightgbm_and_ngboost_and_rf():
    boston = load_boston()
    y = boston.target
    x = boston.data

    # 通过2次train_test_split划分test  train  validation数据集
    X_intermediate, X_test, y_intermediate, y_test = train_test_split(
        x, y, shuffle=True, test_size=0.2, random_state=1)

    # train/validation split (gives us train and validation sets)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_intermediate,
        y_intermediate,
        shuffle=False,
        test_size=0.25,
        random_state=1)

    # delete intermediate variables
    del X_intermediate, y_intermediate

    # 数据集的占比
    # print proportions
    print('train: {}% | validation: {}% | test {}%'.format(
        round(len(y_train) / len(y), 2), round(len(y_validation) / len(y), 2),
        round(len(y_test) / len(y), 2)))

    # predict by NGBoost==================================================

    starttime = datetime.datetime.now()
    ngb = NGBRegressor().fit(X_train, y_train)
    Y_preds = ngb.predict(X_test)
    # Y_dists = ngb.pred_dist(X_test)

    # test Mean Squared Error
    test_MSE_ngb = mean_squared_error(Y_preds, y_test)
    print('Test MSE_ngb', test_MSE_ngb)
    endtime = datetime.datetime.now()
    process_time_ngb = endtime - starttime
    print("ngb程序执行时间（秒）={}".format(process_time_ngb))

    # test Negative Log Likelihood
    # 负对数拟然   反应模型的拟合程度  值越小越好
    # test_NLL_ngb = -Y_dists.logpdf(y_test).mean()
    # 暂时不使用，因为其他模型没有NLL 负对数拟然这个值
    # print('Test NLL_ngb', test_NLL_ngb)

    # predict by Random Forest==============================================
    # 在使用默认参数情况下，rf和ngb的mse结果很接近
    starttime = datetime.datetime.now()
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    Y_preds = regr.predict(X_test)
    test_MSE_rf = mean_squared_error(Y_preds, y_test)
    print('Test MSE_rf', test_MSE_rf)
    endtime = datetime.datetime.now()
    process_time_rf = endtime - starttime
    print("rf程序执行时间（秒）={}".format(process_time_rf))

    # predict by LightGBM==============================================
    starttime = datetime.datetime.now()
    lightgbm = lgb.LGBMRegressor()
    lightgbm.fit(X_train, y_train)
    Y_preds = lightgbm.predict(X_test)
    test_MSE_lgb = mean_squared_error(Y_preds, y_test)
    print('Test MSE_lgb', test_MSE_lgb)
    endtime = datetime.datetime.now()
    process_time_lgb = endtime - starttime
    print("lgb程序执行时间（秒）={}".format(process_time_lgb))

    data_summarize = {
        'model': [
            'LightGBM', 'LightGBM', 'Randomforest', 'Randomforest', 'NGBoost',
            'NGBoost'
        ],
        'score': [
            test_MSE_lgb, process_time_lgb, test_MSE_rf, process_time_rf,
            test_MSE_ngb, process_time_ngb
        ],
        'description': [
            'MSE score', 'process_time', 'MSE score', 'process_time',
            'MSE score', 'process_time'
        ]
    }

    df = pd.DataFrame(data_summarize)
    # 为了显示时间更便于查看，每个时间都扩大5倍
    for i in range(len(df.index)):
        if type(df.iloc[i]['score']) is datetime.timedelta:
            df.iloc[i]['score'] = df.iloc[i]['score'].total_seconds() * 5

    print(df)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='model', y='score', hue='description', data=df)
    plt.legend(loc='upper left')
    plt.xlabel("model category")
    plt.ylabel("Score(values)")
    plt.title("Randomforest，LightGBM，NGBoost三种模型在默认参数下的MSE分数和处理时间，分数越小越好")
    plt.xticks(rotation=330)
    plt.show()


# 在-10 到10 的范围里面找到最小的值，一共进行100次尝试
def fmin():
    from hyperopt import fmin, tpe, hp
    best = fmin(fn=lambda x: x**2,
                space=hp.uniform('x', -10, 10),
                algo=tpe.suggest,
                max_evals=10)
    print(best)


# 通过fmin方法求最小值
def fmin_V1():
    from hyperopt import fmin, tpe, hp, STATUS_OK

    def objective(x):
        return {'loss': x**2, 'status': STATUS_OK}

    best = fmin(objective,
                space=hp.uniform('x', -10, 10),
                algo=tpe.suggest,
                max_evals=100)

    print(best)


def fmin_V2():
    import pickle
    import time
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    default_param = {'b': 1}

    # 通过trials可以返回所有的运算结果值
    def objective(params):
        # 一开始代入一个参数x
        print("params value is：", params)
        # 然后更新了params加入了默认参数b
        params.update(default_param)
        # 显示所有的参数，现在已经有2个值了 x和b
        print("params updated value is：", params)
        return {
            # 使用x和b值
            'loss': params['x']**2 + params['b'],
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'other_stuff': {
                'type': None,
                'value': [0, 1, 2]
            },
            # -- attachments are handled differently
            'attachments': {
                'time_module': pickle.dumps(time.time)
            }
        }

    trials = Trials()
    best = fmin(objective,
                space={'x': hp.uniform('x', -10, 10)},
                algo=tpe.suggest,
                max_evals=2,
                trials=trials)

    print(best)
    # 获取trials里面的所有值输出
    print(trials.trials)


def pyll_example():
    import hyperopt.pyll
    from hyperopt.pyll import scope
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    @scope.define  # 这句话必须要有，否则显示没有foo属性
    def foo(a, b=0):
        # 显示a，b的值大小
        print('runing foo a={},b={}'.format(a, b))
        return a + b / 2

    # -- this will print 0, foo is called as usual.
    print(foo(0))

    # 検索スペースの説明では、普通のPythonのように `foo`を使うことができます。
    # これらの2つの呼び出しは実際にはfooを呼び出さず、
    # グラフを評価するためにfooを呼び出す必要があることだけを記録します。

    space1 = scope.foo(hp.uniform('a', 0, 10))
    space2 = scope.foo(hp.uniform('a', 0, 10), hp.normal('b', 0, 1))

    # -- this will print an pyll.Apply node

    # print("space1=", space1)
    # -- this will draw a sample by running foo()
    # print(hyperopt.pyll.stochastic.sample(space1))
    print(hyperopt.pyll.stochastic.sample(space2))


if __name__ == "__main__":
    # 每个子程序例子可以单独执行，打开注释就可以
    lightgbm_and_ngboost_and_rf()
    # fmin()
    # fmin_V1()
    # fmin_V2()
    # pyll_example()
