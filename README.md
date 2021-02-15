### 关于 NGBoost 模型以及 GBDT 相关模型的案例，并包括 SHAP 模型解释,同时包括模型超参数调试对比

### 程序文件结构

```

│  NGBoost_LightGBM_Randomforest.py
│  NGBoost_original.py
│  Ngboost_Regressor_without_tuning.py      NGBoost不使用超参数调整
│  Ngboost_Regressor_with_tuning.py         NGBoost使用hyperOPT超参数调整进行超参数优化
│  NGBoost带超参数调整和不带调整的测试结论.txt
│  README.md
│
├─chart
│      Figure_100.png
│      Figure_200.png
│      Figure_Feature importance plot for distribution parameters.png
│      Figure_Feature_Importance_shap.png
│      Figure_lr.png
│      Figure_Randomforest，LightGBM，NGBoost三种模型在默认参数下的MSE分数和处理时间，分数越小越好.png
│      Figure_越靠近斜线说明拟合效果越好.png
│
├─document
│      GBDT算法原理以及实例理解.txt
│      shap模型解释.txt
│      超参数优化.txt
│
├─module
│      .gitkeep
│      ngb_Normal.p
│
└─optuna
        LightGBM_with_optuna.py      LightGBM使用超参数调整
        optuna_sample.py             optuna超参数调整的一个简单案例

```

### 参考资料：

```
optuna 结果可视化
https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html

optuna 带CV对lightgbm进行超参数调整
https://optuna.readthedocs.io/zh_CN/latest/reference/generated/optuna.integration.lightgbm.LightGBMTunerCV.html

```

### 结论：

```
LightGBM + optuna 超参数调整 可以获得较好的精度和较短的训练时间
NGBoost + hyperOPT 超参数调整 可以获得较好的精度和较短的训练时间
RandomForest + randomsearch超参数调整 可以获得较好的精度和较短的训练时间
RandomForest + bo-tpe超参数方法也可以尝试
```
