from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform, norm, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import xgboost as xgb



def get_linear_model():
    """output a linear classifier with randomised parameter search over nested 3-fold CV

    returns:
    model: sklearn estimator
    """

    ss = StandardScaler()
    lr = LogisticRegression(penalty='l2', max_iter=1000, class_weight=None)  # ridge

    lr_model = Pipeline(steps=(['scale', ss], ['clf', lr]))                # pipeline

    lr_model_params = {
            'clf__C':loguniform(1e-3,1e3)
    }

    # model: classifier with randomised parameter search over nested 3-fold CV
    linear_model = RandomizedSearchCV(lr_model, lr_model_params, n_iter=100, cv=3)

    return clone(linear_model)

def get_nonlinear_model():
    """output a nonlinear SVM classifier with randomised parameter search over nested 3-fold CV

    returns:
    model: sklearn estimator
    """
    ss = StandardScaler()
    svm = SVC(kernel='rbf', probability=True, random_state=42) # kernel SVM

    svm_model = Pipeline(steps=(['scale', ss], ['clf', svm]))

    svm_model_params = {
        'clf__C':loguniform(1e-3,1e3),
        'clf__gamma':loguniform(1e-4,1e1)
    }

    # model: classifier with randomised parameter search over nested 3-fold CV
    nonlinear_model = RandomizedSearchCV(svm_model, svm_model_params, n_iter=100, cv=3)

    return clone(nonlinear_model)

def get_ensemble_model():
    """output a nonlinear XGBoost classifier with randomised parameter search over nested 3-fold CV

    returns:
    model: sklearn estimator
    """
    ss = StandardScaler()
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    xgb_model = Pipeline(steps=(['scale', ss], ['clf', xgb_clf]))

    xgb_model_params = {
        "clf__colsample_bytree": uniform(0.5, 0.5), # default 1
        "clf__gamma": loguniform(1e-1, 1e3),        # default 0
        "clf__learning_rate": uniform(0.03, 0.57),  # default 0.3
        "clf__max_depth": randint(2, 5),            # default 3
        "clf__n_estimators": randint(10, 50),       # default 100
        "clf__subsample": uniform(0.5, 0.25),       # default 1
        "clf__min_child_weight": randint(1, 8)      # default 1
    }

    # model: classifier with randomised parameter search over nested 3-fold CV (more iters to account for large space)
    ensemble_model = RandomizedSearchCV(xgb_model, xgb_model_params, n_iter=250, cv=3)

    return clone(ensemble_model)
