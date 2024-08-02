from analysis_tools import get_metrics, roc_curve, get_tpr_fpr
from collections import defaultdict
import os
import tabulate

def main():
    models = ['dagan', 'faceswap', 'first', 'sadtalker', 'talklip']
    class_scores = defaultdict(list)
    auc_scores = defaultdict(list)
    class_scores, auc_scores = get_classification_scores(models=models, class_scores=class_scores, auc_scores=auc_scores)
    

    metricspath = 'metrics.txt'
    if os.path.exists(metricspath):
        os.remove(metricspath)
    metrics = open(metricspath, 'a')
    metrics.write('Model:Accuracy:F1:Precision:Recall:AUC\n')
    
    og_truths = [0 for i in range(len(class_scores['ogvid']))]

    for model in models:
        truths = [1 for i in range(len(class_scores[model]))] + og_truths
        predictions = class_scores[model] + class_scores['ogvid']
        accuracy, f1, precision, recall = get_metrics(truths, predictions)
        auc = roc_curve(auc_scores['ogvid'], auc_scores[model])
        metrics.write(f'{model:s}:{accuracy:.2f}:{f1:.2f}:{precision:.1f}:{recall:.2f}:{auc:.4f}\n')  
    

def get_classification_scores(models, class_scores, auc_scores):
    scorepath = 'scores.txt'
    if os.path.exists(scorepath):
        os.remove(scorepath)
    
    score_txt = open(scorepath, 'a')

    deepfakedir = '../deepfakedir/'
    ogdir = '../ogdir/'
    threshold = 0.5 #Threshold for determining REAL(0) or FAKE(1)

    for model in models:
        with open(f"{deepfakedir}{model}.txt") as model_file:
            for line in model_file:
                line.strip('\n')
                _, score, _ = line.split(':')
                auc_scores[model].append(float(score))
                if float(score) >= threshold:
                    class_scores[model].append(1)
                else:
                    class_scores[model].append(0)
        score_txt.write(f"{model}:{class_scores[model]}:{auc_scores[model]}\n")
        model_file.close()
    
    for participant in os.listdir(ogdir):
        with open(f"{ogdir}{participant}") as og_file:
            for line in og_file:
                line.strip('\n')
                _, score, _ = line.split(':')
                auc_scores['ogvid'].append(float(score))
                if float(score) >= threshold:
                    class_scores['ogvid'].append(1)
                else:
                    class_scores['ogvid'].append(0)
    score_txt.write(f"ogvid:{class_scores['ogvid']}:{auc_scores['ogvid']}\n")
    score_txt.close()

    return class_scores, auc_scores



if __name__=='__main__':
    main()