"""B__modelClassification.py

Perform classification using NMF data.
Compares performance of three models: logistic reg, SVM and XGBoost

Configuration options as in config.yaml

"""
import glob, os, sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import yaml

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from functions.models import *

def plot_roc_curves(true_labels, scores, fold_ids, ax=None):

    """ plots ROC curve over folds with fold-wise average"""

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,4))

    folds = np.unique(fold_ids)

    mean_tpr = mean_fpr = 0
    base_fpr = np.linspace(0, 1, 101)

    for f in folds:
        fpr, tpr, _ = roc_curve(true_labels[fold_ids==f], scores[fold_ids==f])
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        ax.plot(base_fpr, tpr, lw=1, linestyle='--', color='grey')

        mean_tpr+=tpr

    ax.plot(base_fpr, mean_tpr/len(folds), lw=2, color='darkred')

    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=16)
    ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=16)
    ax.set_ylabel('TPR', fontsize=16)
    ax.set_xlabel('FPR', fontsize=16)

    ax.plot([0,1], [0,1], linestyle=':', color='darkblue')

def main():
    # load parameters
    with open("config.yaml", 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
    print('')
    print('---------------------------------------------------------')
    print('Configuration:')
    print(yaml.dump(cfg, default_flow_style=False, default_style=''))
    print('---------------------------------------------------------')
    print('')

    # set paths
    datapath = cfg['paths']['datapath']
    nmfpath = cfg['paths']['outDir']
    resultspath = cfg['paths']['results']

    # data
    metric = cfg['data']['metric']

    # other params
    preprocessing_params = cfg['preprocessing']
    wins = 'Winsorise' if preprocessing_params['winsorise'] else 'NoWinsorise'
    cbt = 'Combat' if preprocessing_params['combat'] else 'noCombat'
    unorm = 'Norm' if preprocessing_params['unit_norm'] else 'noNorm'

    nmf_params = cfg['nmf']
    init = nmf_params['init']
    comps = nmf_params['init_comps']
    subj = nmf_params['decomp_data']

    if nmf_params['ard']:
        ard = 'ARD'
    else:
        ard = 'noARD'

    subj = nmf_params['decomp_data']

    print('')
    print('---------------------------------------------------------')
    print('processing: {:} data'.format(metric))
    print('---------------------------------------------------------')
    print('')

    # check NMF loadings exist
    metricdatafile = '{:}{:}-nmf-loadings-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.csv'.format(nmfpath, metric, wins, cbt, unorm, subj, init, comps, ard)
    if os.path.exists(metricdatafile):
        metric_data = pd.read_csv(metricdatafile)
    else:
        print('ERROR: no results, please run NMF')
        sys.exit(1)

    # add control category
    metric_data.insert(2, 'CONTROL', [1 if i=='CONTROL' else 0 for i in metric_data.diagnosis ])

    print('')
    print('*********************************************************')
    print('CLASSIFICATION')
    print('*********************************************************')
    print('')
    print('---------------------------------------------------------')
    print('performing cross-validation')
    print('---------------------------------------------------------')
    print('')

    # 5-fold CV - stratified on DX categories
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # all metric columns
    x = metric_data.loc[:, metric_data.columns.str.contains(metric)]
    meta = metric_data.loc[:, np.invert(metric_data.columns.str.contains(metric))]
    # HC vs any DX
    y = np.array(metric_data.CONTROL)

    # begin K-fold
    preds = np.zeros((len(meta), 3))
    probs = np.zeros((len(meta), 3))
    fold = np.zeros((len(meta), 1))
    for n, (train_idx, test_idx) in enumerate(skf.split(metric_data, metric_data.diagnosis)):

        print('FOLD {:}:------------------------------------------------'.format(n+1))
        train_x, test_x = x.iloc[train_idx], x.iloc[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]
        fold[test_idx] = n+1

        for m, (model_name, model) in enumerate(zip(['linear', 'nonlinear', 'ensemble'], [get_linear_model(), get_nonlinear_model(), get_ensemble_model()])):

            print('fitting {:} model'.format(model_name))
            clf = model.fit(train_x, train_y)

            preds[test_idx, m] = clf.predict(test_x)
            probs[test_idx, m] = clf.predict_proba(test_x)[:, 1]

    # collate data
    preds = pd.DataFrame(preds, columns = ['linear_preds', 'nonlinear_preds', 'ensemble_preds'])
    probs = pd.DataFrame(probs, columns = ['linear_probs', 'nonlinear_probs', 'ensemble_probs'])
    fold = pd.DataFrame(fold.astype(int), columns=['fold'])
    predictions = pd.concat((meta, fold, preds, probs), axis=1)

    # accuracies and AUC
    n_fold = len(np.unique(predictions.fold))
    models = ['linear', 'nonlinear', 'ensemble']

    fold_accuracy = np.zeros((n_fold, len(models)))
    fold_auc = np.zeros((n_fold, len(models)))

    for n, f in enumerate(np.unique(predictions.fold)):
        for m, model in enumerate(models):
            fold_accuracy[n, m] = accuracy_score(predictions.CONTROL[predictions.fold==f], predictions[model+'_preds'][predictions.fold==f])
            fold_auc[n, m] = roc_auc_score(predictions.CONTROL[predictions.fold==f], predictions[model+'_preds'][predictions.fold==f])

    fold_accuracy = pd.DataFrame(fold_accuracy, columns=models)
    fold_accuracy.insert(0, 'fold', np.unique(predictions.fold))

    fold_auc = pd.DataFrame(fold_auc, columns=models)
    fold_auc.insert(0, 'fold', np.unique(predictions.fold))

    # save out
    print('---------------------------------------------------------')
    print('saving model predictions loadings to: {:}{:}-model_predictions-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.csv'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard))
    predictions.to_csv('{:}{:}-model_predictions-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.csv'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard), index=False)
    print('')
    print('saving model accuracies loadings to: {:}{:}-fold_accuracies-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.csv'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard))
    fold_accuracy.to_csv('{:}{:}-fold_accuracies-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.csv'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard), index=False)
    print('saving model ROC AUC loadings to: {:}{:}-fold_auc-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.csv'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard))
    fold_auc.to_csv('{:}{:}-fold_auc-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.csv'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard), index=False)
    print('')
    print('---------------------------------------------------------')

    # PLOT ROC CURVES
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,3), sharey=True)

    plot_roc_curves(predictions.CONTROL.values, predictions.nonlinear_probs.values, predictions.fold.values, ax=ax1)
    plot_roc_curves(predictions.CONTROL.values, predictions.linear_probs.values, predictions.fold.values, ax=ax2)
    plot_roc_curves(predictions.CONTROL.values, predictions.ensemble_probs.values, predictions.fold.values, ax=ax3)

    ax1.set_title('SVM')
    ax2.set_title('LogR')
    ax3.set_title('XGB')

    print('---------------------------------------------------------')
    print('saving roc curves to {:}{:}-roc_curves-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.png'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard))
    plt.tight_layout()
    plt.savefig('{:}{:}-roc_curves-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.png'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard))

    # PLOT ACCURACIES AND AUCS
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4), sharey=False)

    for dat, val, ax, pal in zip([fold_accuracy, fold_auc], ['accuracy', 'auc'], [ax1, ax2], ['Blues', 'Reds']):
        plot_df = dat.melt(id_vars='fold', var_name='model', value_name=val)

        sns.boxplot(x='model', y=val, data=plot_df,  palette=pal, ax=ax, )

        plt.ylim(0,1)
        ax.axhline(0.5, linestyle='-', lw=1, color='grey')
        ax.set_yticks([0,0.5,1])
        ax.set_yticklabels([0,0.5,1], fontsize=18)
        ax.set_xticklabels(['LogR', 'SVM', 'XGBoost'], fontsize=18)
        ax.set_xlabel('')
        ax.set_ylabel(val, fontsize=18)

        sns.despine(top=False, right=False)

    print('saving roc curves to {:}{:}-model_accuracies-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.png'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard))
    plt.tight_layout()
    plt.savefig('{:}{:}-model_accuracies-{:}-{:}-{:}-{:}-init{:}-{:}comp-{:}.png'.format(resultspath, metric, wins, cbt, unorm, subj, init, comps, ard))
    print('')
    print('---------------------------------------------------------')

if __name__ == '__main__':
    main()
