import my_constants
from confidence_interval import bootstrap_ci
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
from matplotlib import pyplot
# from skmisc.loess import loess
from prediction import predict_classification
# from loess.loess_1d import loess_1d
import statsmodels.nonparametric.smoothers_lowess
import re
import os
from joblib import load
from prediction import predict

pyplot.rcParams['font.size'] = 14

def performance_metrics(y_true, y_pred, save_results, cutoffs, filename_prefix=''):
    performance_metrics_dict = {
        my_constants.PERCENT_CUTOFF: [],
        # "Cutoff": [],
        my_constants.SENSITIVITY: [],
        my_constants.SPECIFICITY: [],
        my_constants.PPV: [],
        my_constants.NPV: [],
        my_constants.LIFT: []
    }
    auc_ci = bootstrap_ci(y_true, y_pred, roc_auc_score)
    print("AUC: {:.2f} ({:.2f}, {:.2f})".format(calculate_auc(y_true, y_pred), auc_ci[0], auc_ci[1]))
    # plot_roc(y_true, y_pred, filename_prefix)
    bootstrap_roc(y_true, y_pred, n_bootstraps=500, filename_prefix=filename_prefix)
    plot_precision_recall_curve(y_true, y_pred, filename_prefix)

    for percentile in cutoffs:
        binary_y_pred, cutoff = predict_classification(y_pred, percentile)
        performance_metric_for_cutoff(y_true, binary_y_pred, percentile)

    if save_results:
        df = pd.DataFrame(data=performance_metrics_dict)
        filename = filename_prefix + "performance_metrics.csv"
        df.to_csv(filename)
        print(f'File saved: {os.path.abspath(filename)}')
    smooth_calibration_plot(y_true, y_pred, save_results, filename_prefix)
    return performance_metrics_dict


def performance_metric_for_cutoff(y_true, binary_y_pred, percent_cutoff, metrics=['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Lift'], performance_metrics_dict=None):
    if not performance_metrics_dict:
        performance_metrics_dict = {
            my_constants.PPV:[]
        }
        for m in metrics:
            performance_metrics_dict[m] = []
    performance_metrics_dict[my_constants.PERCENT_CUTOFF] = percent_cutoff
    # performance_metrics_dict["Cutoff"].append("{:.2f}".format(percent_cutoff))
    print("Top " + str(percent_cutoff) + "% of the predictions")

    # PPV - we want it anyway to measure overall utility
    ppv = precision_score(y_true, binary_y_pred)
    ppv_ci = bootstrap_ci(y_true, binary_y_pred, precision_score)
    ppv_formatted = format_score(ppv, ppv_ci)
    performance_metrics_dict[my_constants.PPV].append(ppv_formatted)
    if my_constants.SENSITIVITY in metrics:
        recall = recall_score(y_true, binary_y_pred)
        recall_ci = bootstrap_ci(y_true, binary_y_pred, recall_score)
        recall_formatted = format_score(recall, recall_ci)
        performance_metrics_dict[my_constants.SENSITIVITY].append(recall_formatted)
        print("Sensitivity (Recall): " + recall_formatted)
    if my_constants.SPECIFICITY in metrics:
        specificity = specificity_score(y_true, binary_y_pred)
        specificity_ci = bootstrap_ci(y_true, binary_y_pred, specificity_score)
        specificity_formatted = format_score(specificity, specificity_ci)
        performance_metrics_dict[my_constants.SPECIFICITY].append(specificity_formatted)
        # print("Specificity: " + specificity_formatted)
    # if 'PPV' in metrics:
    #     ppv = precision_score(y_true, binary_y_pred)
    #     ppv_ci = bootstrap_ci(y_true, binary_y_pred, precision_score)
    #     ppv_formatted = format_score(ppv, ppv_ci)
    #     performance_metrics_dict["PPV"].append(ppv_formatted)
        # print("Positive Predictive Value (Precision): " + ppv_formatted)
    if my_constants.NPV in metrics:
        npv = npv_score(y_true, binary_y_pred)
        npv_ci = bootstrap_ci(y_true, binary_y_pred, npv_score)
        npv_formatted = format_score(npv, npv_ci)
        performance_metrics_dict[my_constants.NPV].append(npv_formatted)
        # print("Negative Predictive Value: " + npv_formatted)
    if my_constants.LIFT in metrics:
        lift = lift_score(np.array(y_true).ravel(), binary_y_pred)
        lift_ci = bootstrap_ci(y_true, binary_y_pred, lift_score)
        lift_formatted = format_score(lift, lift_ci)
        performance_metrics_dict[my_constants.LIFT].append(lift_formatted)
        # print("Lift: " + lift_formatted)
    # print("~~~~~~")
    return performance_metrics_dict


def smooth_calibration_plot(y_true, y_pred, save_results=False, filename_prefix=''):

    y_true = y_true.iloc[:, 0]

    line = np.arange(0, 1.1, 0.1)
    pyplot.plot(line, line, color='r')

    pyplot.scatter(y_pred, y_true, alpha=0.1)

    # sort = np.argsort(y_pred)
    # y_pred_sorted = y_pred[sort]
    # y_true_sorted = np.array(y_true)[sort]
    # # try:  # TODO
    # #     y_true_sorted = np.array(y_true['death'])[sort]
    # # except KeyError:
    # #     y_true_sorted = np.array(y_true['hospital_death'])[sort]
    # y_pred_unique, indices = np.unique(y_pred_sorted, return_index=True)
    # y_true_unique = y_true_sorted[indices]

    # loess_smooth = loess(y_pred_sorted, y_true_sorted)
    # loess_smooth.fit()
    # loess_out = loess_smooth.outputs.fitted_values
    # pyplot.plot(y_pred_unique, loess_out)

    # xout, yout, wout = loess_1d(y_pred_sorted, y_true_sorted)
    # pyplot.plot(xout, yout)

    out = statsmodels.nonparametric.smoothers_lowess.lowess(
        np.array(y_true),
        np.array(y_pred),
        is_sorted=False,
        return_sorted=True,
        it=0,
        delta=0.0,
        xvals=None
    )
    pyplot.plot(out[:, 0], out[:, 1])

    discrete_calibration_true, discrete_calibration_pred = calibration_curve(y_true, y_pred, strategy='quantile', n_bins=10)
    pyplot.scatter(discrete_calibration_pred, discrete_calibration_true)
    pyplot.xlabel('Prediction')
    pyplot.ylabel('True Label')
    pyplot.title('Calibration')

    if save_results:
        filename = filename_prefix + "calibration.png"
        pyplot.savefig(filename)
        print(f'Graph saved: {os.path.abspath(filename)}')
    else:
        pyplot.show()


def calculate_auc(y, predicted_y):
    y_true = np.array(y)
    auc = roc_auc_score(y_true.ravel(), predicted_y)
    # print("AUC = " + str(auc))
    return auc


def plot_roc(y_true, y_pred, filename_prefix=''):
    line = np.arange(0, 1.1, 0.1)
    pyplot.plot(line, line, color='r')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # ci = bootstrap_ci(y_true, y_pred, roc_curve)
    pyplot.scatter(fpr, tpr)
    pyplot.title('ROC Curve')
    pyplot.xlabel('False Positive')
    pyplot.ylabel('True Positive')
    filename = filename_prefix + "roc.png"
    pyplot.savefig(filename)
    print(f'Graph saved: {os.path.abspath(filename)}')
    pyplot.show()


def bootstrap_roc(y_true, y_pred, n_bootstraps=1000, filename_prefix=''):
    # Initialize arrays to store bootstrapped ROC data
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    # Generate bootstrapped samples and calculate ROC curve for each
    for i in range(n_bootstraps):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        bootstrapped_true = np.array(y_true).ravel()[indices]
        bootstrapped_pred = np.array(y_pred)[indices]
        fpr, tpr, _ = roc_curve(bootstrapped_true, bootstrapped_pred)
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tprs.append(tpr_interp)

    # Compute mean of bootstrapped ROC curves
    mean_tpr = np.mean(tprs, axis=0)

    # Compute confidence intervals
    ci_low = np.percentile(tprs, 2.5, axis=0)
    ci_high = np.percentile(tprs, 97.5, axis=0)

    # Plot mean ROC curve with confidence intervals
    pyplot.plot(base_fpr, mean_tpr, 'b', label='Mean ROC')
    pyplot.fill_between(base_fpr, ci_low, ci_high, color='grey', alpha=0.3, label='95% CI')
    pyplot.plot([0, 1], [0, 1], 'r--')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC Curve')
    pyplot.legend(loc='lower right')

    # Save plot
    filename = filename_prefix + "roc_with_ci.png"
    pyplot.savefig(filename)
    print(f'Graph saved: {os.path.abspath(filename)}')
    pyplot.show()
    pyplot.close()


def compare_roc(y_dict, n_bootstraps=1000, filename_prefix=''):
    # Initialize figure for subplots
    fig, axs = pyplot.subplots(1, 2, figsize=(10, 5))

    # Function to calculate bootstrapped ROC curve
    def bootstrap_roc(ax, y_true, y_pred, model_title):
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

        for i in range(n_bootstraps):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            bootstrapped_true = np.array(y_true).ravel()[indices]
            bootstrapped_pred = np.array(y_pred)[indices]
            fpr, tpr, _ = roc_curve(bootstrapped_true, bootstrapped_pred)
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tprs.append(tpr_interp)

        mean_tpr = np.mean(tprs, axis=0)
        ci_low = np.percentile(tprs, 2.5, axis=0)
        ci_high = np.percentile(tprs, 97.5, axis=0)

        ax.plot(base_fpr, mean_tpr, 'b', label='Mean ROC')
        ax.fill_between(base_fpr, ci_low, ci_high, color='grey', alpha=0.3, label='95% CI')
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve ({model_title})')
        ax.legend(loc='lower right')

    for i, model in enumerate(y_dict):
        y_true = y_dict[model][0]
        y_pred = y_dict[model][1]
        bootstrap_roc(axs[i], y_true, y_pred, model)

    # Adjust layout
    pyplot.tight_layout()

    # Save plot
    filename = filename_prefix + "roc.png"
    pyplot.savefig(filename)
    print(f'Graphs saved: {os.path.abspath(filename)}')

    # Show plot
    pyplot.show()


def plot_precision_recall_curve(y_true, y_pred, filename_prefix=''):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pyplot.plot(recall, precision)
    pyplot.title('Precision-Recall Curve')
    filename = filename_prefix + "precision_recall.png"
    pyplot.savefig(filename)
    print(f'Graph saved: {os.path.abspath(filename)}')
    pyplot.show()


def npv_score(y_true, y_pred):
    # true_negative, false_positive, false_negative, true_positive = calc_confusion_matrix(y_true, y_pred)
    # return true_negative/(true_negative + false_negative)
    y_true_inverse = (~y_true.astype(bool)).astype(int)
    y_pred_inverse = (~y_pred.astype(bool)).astype(float)
    score = precision_score(y_true_inverse, y_pred_inverse, zero_division=0)
    return score


def ppv_score(y_true, y_pred):
    score = precision_score(y_true, y_pred, zero_division=0)
    return score


def lift_score(y_true, y_pred):
    u, counts = np.unique(y_true, return_counts=True)
    prevalence = counts[1]/len(y_true)
    ppv = precision_score(y_true, y_pred)
    return ppv/prevalence


def specificity_score(y_true, y_pred):
    # true_negative, false_positive, false_negative, true_positive = calc_confusion_matrix(y_true, y_pred)
    y_true_inverse = (~y_true.astype(bool)).astype(int)
    y_pred_inverse = (~y_pred.astype(bool)).astype(float)
    score = recall_score(y_true_inverse, y_pred_inverse,  zero_division=0)
    # return true_negative/(true_negative + false_positive)
    return score


def sensitivity_score(y_true, y_pred):
    score = recall_score(y_true, y_pred, zero_division=0)
    return score


def sensitivity_ci(y_true, y_pred):
    bounds = bootstrap_ci(y_true, y_pred, recall_score)
    return bounds


def ppv_ci(y_true, y_pred):
    bounds = bootstrap_ci(y_true, y_pred, precision_score)
    return bounds


def specificity_ci(y_true, y_pred):
    bounds = bootstrap_ci(y_true, y_pred, specificity_score)
    return bounds


def npv_ci(y_true, y_pred):
    bounds = bootstrap_ci(y_true, y_pred, npv_score)
    return bounds


def calc_confusion_matrix(y_true, y_pred):
    try:
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, y_pred).ravel()
    except:
        num_of_patients = len(y_true)
        if np.array_equal(np.array([0] * num_of_patients), y_true):
            true_negative = np.count_nonzero(y_pred == 0)
            false_positive = np.count_nonzero(y_pred == 1)
            true_positive = 0.0
            false_negative = 0.0
        else:
            true_positive = np.count_nonzero(y_pred == 1)
            false_negative = np.count_nonzero(y_pred == 0)
            true_negative = 0.0
            false_positive = 0.0
    return true_negative, false_positive, false_negative, true_positive


def extract_values(metric_str):
    # Extracts the numerical values from the performance metrics string
    match = re.match(r'([\d.]+)\s*\(([\d.]+),\s*([\d.]+)\)', metric_str)
    if match:
        metric_value = float(match.group(1))
        upper_value = float(match.group(2))
        lower_value = float(match.group(3))
        return metric_value, upper_value, lower_value
    else:
        return None


def format_score(score, ci):
    return "{:.2f} ({:.2f}, {:.2f})".format(score, ci[0], ci[1])

if __name__ == '__main__':
    dataset = my_constants.MIMIC
    trained_model_mimic = load('small_logistic_regression_' + dataset + '.joblib')
    X_test_mimic = pd.read_csv(f'x_test_{dataset}.csv', index_col=[0])
    y_test_mimic = pd.read_csv(f'y_test_{dataset}.csv', index_col=[0])
    y_pred_mimic = predict(trained_model_mimic, X_test_mimic)
    y_dict = {'MIMIC': [y_test_mimic, y_pred_mimic]}
    dataset = my_constants.KAGGLE
    trained_model_kaggle = load('small_logistic_regression_' + dataset + '.joblib')
    X_test_kaggle = pd.read_csv(f'x_test_{dataset}.csv', index_col=[0])
    y_test_kaggle = pd.read_csv(f'y_test_{dataset}.csv', index_col=[0])
    y_pred_kaggle = predict(trained_model_kaggle, X_test_kaggle)
    y_dict['GOSSIS'] = [y_test_kaggle, y_pred_kaggle]
    compare_roc(y_dict)