
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



def ml():
    with open('rawdata/predata.pkl', 'rb') as f:
        predata = pickle.load(f)

    data, clin = predata['data'], predata['clin']
    rizvi_data = data.loc[clin.dataset.values == 'd1', :].values
    rizvi_resp = clin.resp.values[clin.dataset.values == 'd1']
    miao_data = data.loc[clin.dataset.values != 'd1', :].values
    miao_resp = clin.resp.values[clin.dataset.values != 'd1']

    with open('hyper_params_search/best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    seeds = params['seeds']

    trainx, validx, trainy, validy = train_test_split(
        rizvi_data, rizvi_resp, test_size=0.1, shuffle=True,
        random_state=seeds, stratify=rizvi_resp
    )

    # Logistic Regression
    Cs = [1., 2., 3., 4.]
    valid_aucs = []
    for C in Cs:
        logit = LogisticRegression(
            n_jobs=50, random_state=seeds, C=C
        )
        logit.fit(trainx, trainy)
        valid_pred = logit.predict_proba(validx)
        valid_aucs.append(roc_auc_score(validy, valid_pred[:, 1]))

    C = Cs[np.argmax(valid_aucs)]
    print('BestLogitC: {}'.format(C))
    # C=2
    logit = LogisticRegression(n_jobs=50, random_state=seeds, C=C)
    logit.fit(trainx, trainy)
    logit_pred = logit.predict_proba(miao_data)
    logit_auc = roc_auc_score(miao_resp, logit_pred[:, 1])
    print('Logit AUROC: {:.4f}'.format(logit_auc))

    # Random Forest
    n_trees = [100, 500, 1000, 2000]
    valid_aucs = []
    for n_tree in n_trees:
        rf = RandomForestClassifier(
            n_estimators=n_tree, n_jobs=50, random_state=seeds
        )
        rf.fit(trainx, trainy)
        valid_pred = rf.predict_proba(validx)
        valid_aucs.append(roc_auc_score(validy, valid_pred[:, 1]))

    n_tree = n_trees[np.argmax(valid_aucs)]
    print('BestRFTreeNum: {}'.format(n_tree))
    # 100
    rf = RandomForestClassifier(n_estimators=n_tree, n_jobs=50, random_state=seeds)
    rf.fit(trainx, trainy)
    rf_pred = rf.predict_proba(miao_data)
    rf_auc = roc_auc_score(miao_resp, rf_pred[:, 1])
    print('RF AUROC: {:.4f}'.format(rf_auc))

    # Rbf-SVM
    Cs = [1., 2., 3., 4.]
    valid_aucs = []
    for C in Cs:
        rbfsvm = SVC(
            kernel='rbf', C=C, probability=True, random_state=seeds
        )
        rbfsvm.fit(trainx, trainy)
        valid_pred = rbfsvm.predict_proba(validx)
        valid_aucs.append(roc_auc_score(validy, valid_pred[:, 1]))

    C = Cs[np.argmax(valid_aucs)]
    print('BestRbfSVMC: {}'.format(C))
    # C=4
    rbfsvm = SVC(
        kernel='rbf', C=C, probability=True, random_state=seeds
    )
    rbfsvm.fit(trainx, trainy)
    rbfsvm_pred = rbfsvm.predict_proba(miao_data)
    rbfsvm_auc = roc_auc_score(miao_resp, rbfsvm_pred[:, 1])
    print('rbf-SVM AUROC: {:.4f}'.format(rbfsvm_auc))

    # Linear-SVM
    Cs = [1., 2., 3., 4.]
    valid_aucs = []
    for C in Cs:
        linearsvm = SVC(
            kernel='linear', C=C, probability=True, random_state=seeds
        )
        linearsvm.fit(trainx, trainy)
        valid_pred = linearsvm.predict_proba(validx)
        valid_aucs.append(roc_auc_score(validy, valid_pred[:, 1]))

    C = Cs[np.argmax(valid_aucs)]
    print('BestLinearSVMC: {}'.format(C))
    # C=2
    linearsvm = SVC(
        kernel='linear', C=C, probability=True, random_state=seeds
    )
    linearsvm.fit(trainx, trainy)
    linearsvm_pred = linearsvm.predict_proba(miao_data)
    linearsvm_auc = roc_auc_score(miao_resp, linearsvm_pred[:, 1])
    print('linear-SVM AUROC: {:.4f}'.format(linearsvm_auc))

    # Xgboost
    params = ParameterGrid({
        'max_depth': [2, 4, 6], 'eta': [0.01, 0.1, 0.3],
        'objective': ['binary:logistic'], 'eval_metric': ['auc']
    })
    dtrain = xgb.DMatrix(data=trainx, label=trainy)
    best_valid_auc = 0.
    best_param = None
    for param in list(params):
        bst = xgb.train(params=param, dtrain=dtrain)
        dvalid = xgb.DMatrix(data=validx)
        valid_pred = bst.predict(dvalid)
        valid_auc = roc_auc_score(validy, valid_pred)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_param = param

    print('XgBoost Params')
    print(best_param)
    # 'eta': 0.3, 'eval_metric': 'auc', 'max_depth': 2,
    # 'objective': 'binary:logistic'
    bst = xgb.train(params=best_param, dtrain=dtrain)
    dtest = xgb.DMatrix(miao_data)
    xgb_pred = bst.predict(dtest)
    xgb_auc = roc_auc_score(miao_resp, xgb_pred)
    print('XGBoost AUC: {:.4f}'.format(xgb_auc))

    # total predict data
    pred_data = pd.DataFrame({
        'ytrue': miao_resp, 'logit': logit_pred[:, 1], 'rf': rf_pred[:, 1],
        'rbfsvm': rbfsvm_pred[:, 1], 'linearsvm': linearsvm_pred[:, 1],
        'xgb': xgb_pred
    })
    pred_data.to_csv('hyper_params_search/ml_pred_result.csv', index=None)



if __name__ == '__main__':
    ml()

