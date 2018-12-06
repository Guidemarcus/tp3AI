from SoftmaxClassifier import SoftmaxClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss

    nb_run = 3

    models = [
        SoftmaxClassifier(),
        RandomForestClassifier(),
        DecisionTreeClassifier()
    ]

    scoring = ['neg_log_loss', 'precision_macro','recall_macro','f1_macro']
    print(compare(models,X_train,y_train_label,nb_run,scoring))