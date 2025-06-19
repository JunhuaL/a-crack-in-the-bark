import pandas as pd
import numpy as np
import argparse
from sklearn import metrics
import matplotlib.pyplot as plt

def precision_at_base_rate(result_df,set_tpr,base_rate):
    tpr = result_df['TPR'].values
    fpr = result_df['FPR'].values
    fpr_at_set_tpr = fpr[np.where(tpr>=set_tpr)[0][0]]
    ppv = (base_rate*set_tpr)/((base_rate*set_tpr) + ((1-base_rate)*fpr_at_set_tpr) + 0.0001)
    return ppv

def recall_at_base_rate(result_df,set_tpr,base_rate):
    tpr = result_df['TPR'].values
    fnr = 1 - result_df['TPR'].values
    fnr_at_set_tpr = fnr[np.where(tpr>=set_tpr)[0][0]]
    ppv = (base_rate*set_tpr)/((base_rate*set_tpr) + ((1-base_rate)*fnr_at_set_tpr) + 0.0001)
    return ppv

def main(args):
    scores_df = pd.read_csv(args.table_path)
    results = pd.DataFrame()
    preds = pd.concat([scores_df['adv_score'],scores_df['no_w_score']]).values
    t_labels = [1] * len(scores_df['adv_score']) + [0] * len(scores_df['no_w_score'])
    fpr,tpr, thresholds = metrics.roc_curve(t_labels,-preds,pos_label=1)
    results['TPR'] = tpr
    results['FPR'] = fpr
    results['Threshold'] = thresholds
    results = results.sort_values('Threshold',ascending=False).reset_index(drop=True).drop_duplicates()

    base_rates = np.linspace(0,1,20,endpoint=True)
    set_tprs = [(.99,'d',10), (0.95,'*',14), (0.75,'v',12), (0.5,'s',9)]
    for set_tpr,m,ms in set_tprs:
        precisions = []
        for base_rate in base_rates:
            ppv = precision_at_base_rate(results,set_tpr,base_rate)
            precisions.append(ppv)

        plt.plot(base_rates,precisions,linewidth=3,marker=m,markeredgecolor='white',markersize=ms,label=f'TPR={set_tpr}')
    plt.xlabel('Base Rate',horizontalalignment='left')
    plt.ylabel('Precision',verticalalignment='bottom')
    plt.legend(['TPR=0.99','TPR=0.95','TPR=0.75','TPR=0.5'],loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precision at base rate calculation")
    parser.add_argument("--table_path", type=str)
    args = parser.parse_args()
    main(args)