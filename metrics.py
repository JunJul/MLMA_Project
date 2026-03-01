import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def classification_result(y_true, preds, model_name, saved_path):
    """
    Displays the confusion matrix and classification report.
    Calculates Accuracy, Recall, and Specificity.
    """
    print(f"\n------ {model_name} Test Results ------")
    
    report_text = classification_report(y_true, preds)
    print(report_text)
    
    cm = confusion_matrix(y_true, preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    
    plt.savefig(saved_path / f"{model_name}_cm.png", bbox_inches='tight')
    
    with open(saved_path / f'{model_name}_classification_report.txt', 'w') as f:
        f.write(report_text)
        pass

    pass
