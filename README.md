### 关于 NGBoost 模型以及 GBDT 相关模型的案例，并包括 SHAP 模型解释,同时包括模型超参数调试对比

### 程序文件结构

```
│  NGBoost_LightGBM_Randomforest.py
│  NGBoost_original.py
│  Ngboost_Regressor_without_tuning.py
│  Ngboost_Regressor_with_tuning.py
│  README.md
│  测试结论.txt
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
└─module
        .gitkeep
        ngb_Normal.p
```
