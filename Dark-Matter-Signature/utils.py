import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from scikitplot.metrics import plot_confusion_matrix, plot_roc

def evaluate_performace(y_true, y_pred, threshold=0.5):
    y_probs = np.hstack([1-y_pred.reshape(-1, 1), y_pred.reshape(-1, 1)])
    y_pred = y_pred >= threshold
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))
    print("Precision Score: ", precision_score(y_true, y_pred))
    print("Recall Score: ", recall_score(y_true, y_pred))

    plot_confusion_matrix(y_true, y_pred, normalize=True, cmap="winter")
    plot_roc(y_true, y_probs, classes_to_plot=[0], plot_macro=False, plot_micro=False)
    plt.show()