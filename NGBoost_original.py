#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 下面是官网原始NGBoost代码
# https://stanfordmlgroup.github.io/ngboost/1-useage.html

# 关于shap解释的资料看下面链接
# https://zhuanlan.zhihu.com/p/83412330

from ngboost import NGBRegressor
from ngboost.distns import Exponential, Normal
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from pathlib import Path

# 导入数据
X, Y = load_boston(True)
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=1)
# print(X.shape)
# print(Y.shape)


def ngb_Normal():
    ngb_Normal = NGBRegressor(Dist=Normal).fit(X_train, Y_train)
    globals()['ngb_Normal'] = ngb_Normal
    Y_preds = ngb_Normal.predict(X_test)
    Y_dists = ngb_Normal.pred_dist(X_test)
    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print('Test MSE_Normal', test_MSE)
    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(Y_test).mean()
    print('Test NLL_Normal', test_NLL)


# 默认dist是Normal，使用Dist=Exponential进行测试，发现指数方式结果变差了
# Test MSE_Normal 7.425261043033081
# Test NLL_Normal 3.098725688371598

# Test MSE_Exponential 10.559442720032198
# Test NLL_Exponential 4.039629015827819


def ngb_Exponential():
    print("====================================")
    ngb_Exponential = NGBRegressor(Dist=Exponential).fit(X_train, Y_train)
    globals()['ngb_Exponential'] = ngb_Exponential
    Y_preds = ngb_Exponential.predict(X_test)
    Y_dists = ngb_Exponential.pred_dist(X_test)

    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print('Test MSE_Exponential', test_MSE)
    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(Y_test).mean()
    print('Test NLL_Exponential', test_NLL)


def ngb_cv():
    print("====================================")
    b1 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=2)
    b2 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4)
    param_grid = {'minibatch_frac': [1.0, 0.5], 'Base': [b1, b2]}
    ngb = NGBRegressor(Dist=Normal, verbose=True)
    grid_search = GridSearchCV(ngb, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, Y_train)
    best_params = grid_search.best_params_
    print(best_params)
    ngb_cv = NGBRegressor(Dist=Normal, verbose=True,
                          **best_params).fit(X_train, Y_train)
    globals()['ngb_cv'] = ngb_cv
    Y_preds = ngb_cv.predict(X_test)
    Y_dists = ngb_cv.pred_dist(X_test)
    # test Mean Squared Error
    test_MSE_CV = mean_squared_error(Y_preds, Y_test)
    print('Test MSE_CV', test_MSE_CV)
    # test Negative Log Likelihood
    test_NLL_CV = -Y_dists.logpdf(Y_test).mean()
    print('Test NLL_CV', test_NLL_CV)


def result_summary():
    global ngb_Normal, ngb_Exponential
    # 比较实际值，normal下预测值，exponential下预测值
    print("实际值:", Y_test[0:5])
    print("ngb_Normal:", ngb_Normal.predict(X_test)[0:5])
    print("ngb_Exponential:", ngb_Exponential.predict(X_test)[0:5])
    print("ngb_cv:", ngb_cv.predict(X_test)[0:5])


def feature_importance():
    global ngb_Normal
    ngb_Normal = NGBRegressor(verbose=True).fit(X_train, Y_train)
    # Feature importance for loc trees
    feature_importance_loc = ngb_Normal.feature_importances_[0]
    # Feature importance for scale trees
    feature_importance_scale = ngb_Normal.feature_importances_[1]

    # dataframe制作
    df_loc = pd.DataFrame({
        'feature': load_boston()['feature_names'],
        'importance': feature_importance_loc
    }).sort_values('importance', ascending=False)

    # dataframe制作
    df_scale = pd.DataFrame({
        'feature': load_boston()['feature_names'],
        'importance': feature_importance_scale
    }).sort_values('importance', ascending=False)

    # 通过sns绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Feature importance plot for distribution parameters",
                 fontsize=17)
    sns.barplot(x='importance',
                y='feature',
                ax=ax1,
                data=df_loc,
                color="skyblue").set_title('loc param')
    sns.barplot(x='importance',
                y='feature',
                ax=ax2,
                data=df_scale,
                color="skyblue").set_title('scale param')
    plt.show()


def xgboost_shap():
    import shap
    import xgboost
    # shap.initjs()  # notebook环境下，加载用于可视化的JS代码

    # 我们先训练好一个XGBoost model
    X, y = shap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y),
                          100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值
    # 可视化第一个prediction的解释   如果不想用JS,传入matplotlib=True
    # 如果是在jupyter notebook 上面使用去掉force_plot里面的【matplotlib=True】
    # 如果在python里面显示就需要force_plot里面加入【matplotlib=True】
    shap.force_plot(explainer.expected_value,
                    shap_values[0, :],
                    X.iloc[0, :],
                    matplotlib=True)
    # 取每个特征的SHAP值的绝对值的平均值作为该特征的重要性，得到一个标准的条形图(multi-class则生成堆叠的条形图)
    shap.summary_plot(shap_values, X, plot_type="bar")


def shap_feature_show():
    global ngb_Normal
    # notebook环境下，加载用于可视化的JS代码
    shap.initjs()
    # SHAP plot for loc trees
    explainer = shap.TreeExplainer(
        ngb_Normal, model_output=0)  # use model_output = 1 for scale trees
    # 传入特征矩阵X，计算SHAP值
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values,
                      X_train,
                      feature_names=load_boston()['feature_names'])

    # 可视化第一个prediction的解释   如果不想用JS,传入matplotlib=True
    # 如果是在jupyter notebook 上面使用去掉force_plot里面的【matplotlib=True】
    # 如果在python里面显示就需要force_plot里面加入【matplotlib=True】
    # X_train_df = pandas.DataFrame(X_train)
    # shap.force_plot(explainer.expected_value,
    #                 shap_values[0, :],
    #                 X_train_df.iloc[0, :],
    #                 matplotlib=True)

    # 取每个特征的SHAP值的绝对值的平均值作为该特征的重要性，得到一个标准的条形图(multi-class则生成堆叠的条形图)
    shap.summary_plot(shap_values, X_train, plot_type="bar")


def save_model():
    global ngb_Normal
    file_path = Path(__file__).resolve().parent / 'module' / 'ngb_Normal.p'
    print(file_path)
    with file_path.open("wb") as f:
        pickle.dump(ngb_Normal, f)


def load_model():
    file_path = Path(__file__).resolve().parent / 'module' / 'ngb_Normal.p'
    with file_path.open("rb") as f:
        ngb_unpickled_normal = pickle.load(f)
        globals()['ngb_unpickled_normal'] = ngb_unpickled_normal
    # 利用装载的ngb model来进行预测测试
    Y_preds = ngb_unpickled_normal.predict(X_test)
    Y_dists = ngb_unpickled_normal.pred_dist(X_test)
    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print('Test MSE_Normal_unpickled', test_MSE)
    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(Y_test).mean()
    print('Test NLL_Normal_unpickled', test_NLL)


def main():
    # ngb不同模型的测试结果
    ngb_Normal()
    # ngb_Exponential()
    # ngb_cv()
    # 数据结果摘要显示
    # result_summary()
    # 特征重要度显示
    # feature_importance()
    # xgboost_shap()
    # shap_feature_show()
    # save_model()
    load_model()


if __name__ == "__main__":
    main()
