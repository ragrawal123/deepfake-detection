import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
import os
import pandas as pd

def get_tpr_fpr(original_vid_scores, df_vid_scores, decision_thresh, print_results= False):
    """
    Compute true positive rate and false positive rate for a given decision threshold.
    Here, positive is a detection of a fake video.

    Parameters:
    original_vid_scores (list): scores for original videos
    df_vid_scores (list): scores for deepfaked videos
    decision_thresh (int or float): decision threshold, i.e., 0.5

    Returns:
    tpr (float): true positive rate
    fpr (float): false positive rate
    """
  
    true_positives = 0 
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    false_positive_indices = []
    false_negative_indices = []

    for i in range(len(df_vid_scores)):
        if df_vid_scores[i] > decision_thresh:
            true_positives += 1
        else:
            false_negatives += 1
            false_negative_indices.append(i)
    for i in range(len(original_vid_scores)):
        if original_vid_scores[i] > decision_thresh:
            false_positives += 1
            false_positive_indices.append(i)
        else:
            true_negatives += 1
    if print_results:
        print(f"--------------- Threshold: {decision_thresh} --------------- ")
        print(f"False positives: {false_positives}/{len(original_vid_scores)} original_vid_scores videos.")
        print(f"True positives: {true_positives}/{len(df_vid_scores)} fake videos.")
        print(f"False negatives: {false_negatives}/{len(df_vid_scores)} fake videos.")
        print(f"True negatives: {true_negatives}/{len(original_vid_scores)} original_vid_scores videos.")
    if true_positives + false_negatives == 0:
        if print_results:
            print("TPR: N/A")
        tpr = None
    else:
        tpr = true_positives / (true_positives + false_negatives)
        if print_results:
            print(f"TPR: {tpr}")
    if false_positives + true_negatives == 0:
        if print_results:
            print("FPR: N/A")
        fpr = None
    else:
        fpr = false_positives / (false_positives + true_negatives)
        if print_results:
            print(f"FPR: {fpr}")
    return tpr, fpr


def roc_curve(original_vid_scores, df_vid_scores, model_name):
    """
    Plot ROC curve and compute AUC.

    Parameters:
    original_vid_scores (list): scores for original videos
    df_vid_scores (list): scores for deepfaked videos
    """
    tprs = []
    fprs = []
    #for each possible threshold, get tpr and fpr
    min_t = min(np.min(original_vid_scores), np.min(df_vid_scores))
    max_t = max(np.max(original_vid_scores), np.max(df_vid_scores))
    num_iters = 1000 # how many thresholds to iterate through when making ROC
    iter_ts = np.linspace(min_t, max_t, num_iters)
    for t in iter_ts:
        tpr, fpr = get_tpr_fpr(original_vid_scores, df_vid_scores, t)
        tprs.append(tpr)
        fprs.append(fpr)
    

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    fprs_order = np.argsort(fprs)
    fprs = fprs[fprs_order] #sort so integral has correct sign
    tprs = tprs[fprs_order]

    csv_file = f"./plots/{model_name}/{model_name}rates.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    scores = {'fprs': fprs, 'tprs': tprs}
    values = pd.DataFrame(scores)
    values.to_csv(csv_file, index=False)


    auc = np.trapz(tprs, fprs)
    #print("AUC: ", auc)

    plt.plot(fprs, tprs)
    plt.title(f"AUC {auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #plt.show()
    return auc

def get_metrics(ground_truths, predictions):
    accuracy = balanced_accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions)
    recall = recall_score(ground_truths, predictions)

    return [accuracy, f1, precision, recall]


'''
Implement function to automatically create gnuplots
'''