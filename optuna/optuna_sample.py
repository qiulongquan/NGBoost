import optuna


def objective(trial):
    # 从-100到100之间均匀的返回一个值
    x = trial.suggest_uniform('x', -100, 100)
    y = trial.suggest_categorical('y', [-1, 0, 1])
    return x**2 + y


study = optuna.create_study()
study.optimize(objective, n_trials=10)

# 没有最后的.show()将不会在python里面显示，jupyter里面会显示
optuna.visualization.plot_optimization_history(study).show()

# 结果输出
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
