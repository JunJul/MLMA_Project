import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from dataset import CHEXPERT_CLASSES

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from dataset import CHEXPERT_CLASSES


COMPETITION_TASKS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

def classification_result(y_true, y_probs, model_name, saved_path):
    """
    Displays the multi-label classification report.
    Calculates AUROC, plots ROC curves, and saves all metrics to a text file.
    
    y_true: Numpy array of shape (N, 14) with ground truth 0s and 1s
    y_probs: Numpy array of shape (N, 14) with probabilities (after Sigmoid)
    """
    print(f"\n------ {model_name} Test Results ------")
    
    y_preds = (y_probs >= 0.5).astype(int)
    
    report_text = classification_report(
        y_true, 
        y_preds, 
        target_names=CHEXPERT_CLASSES, 
        zero_division=0
    )
    print(report_text)

    fig, ax = plt.subplots(figsize=(10, 8))
    auroc_scores = []
    
    auroc_log = ["\n\n--- Stanford Competition AUROC Scores ---"]

    for task in COMPETITION_TASKS:
        task_idx = CHEXPERT_CLASSES.index(task)
        
        task_true = y_true[:, task_idx]
        task_probs = y_probs[:, task_idx]
        
        try:
            score = roc_auc_score(task_true, task_probs)
            auroc_scores.append(score)

            auroc_log.append(f"{task:>20}: {score:.4f}")
            
            fpr, tpr, _ = roc_curve(task_true, task_probs)
            ax.plot(fpr, tpr, lw=2, label=f"{task} (AUC = {score:.3f})")
            
        except ValueError:
            warning_msg = f"{task:>20}: N/A (No positive samples in test set)"
            print(warning_msg)
            auroc_log.append(warning_msg)

    # Calculate and log the mean AUROC
    if auroc_scores:
        mean_auroc = np.mean(auroc_scores)
        auroc_log.append("-" * 30)
        auroc_log.append(f"{'Mean Competition AUROC':>20}: {mean_auroc:.4f}")
        print(f"\nMean Competition AUROC: {mean_auroc:.4f}\n")

    full_report_text = report_text + "\n".join(auroc_log)
    with open(saved_path / f'{model_name}_classification_report.txt', 'w') as f:
        f.write(full_report_text)

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(f'ROC Curves for Stanford Tasks: {model_name}', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.savefig(saved_path / f"{model_name}_roc_curves.png", bbox_inches='tight')
    plt.close()


