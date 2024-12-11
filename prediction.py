import numpy as np
import data


def predict(model, X):
    preds = model.predict_proba(X)
    return preds[:, 1]


def predict_classification(y_pred, percentile):
    class_pred = y_pred.copy()
    sort = np.argsort(y_pred)[::-1]
    y_pred_sorted = y_pred[sort]
    top = round(percentile / 100 * len(y_pred_sorted))
    cutoff = y_pred_sorted[top]
    class_pred[sort[:top]] = 1
    class_pred[sort[top:]] = 0
    return class_pred, cutoff