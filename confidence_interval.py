from scipy.stats import bootstrap
import numpy as np

def bootstrap_ci(y_true, y_pred, metric, confidence_level=0.95, n_resamples=500,
                 method='percentile', random_state=42):

    def statistic(*indices):
        indices = np.array(indices)[0, :]
        return metric(np.array(y_true).ravel()[indices], np.array(y_pred)[indices])

    indices = (np.arange(len(y_true)), )
    bootstrap_res = bootstrap(indices,
                              statistic=statistic,
                              n_resamples=n_resamples,
                              confidence_level=confidence_level,
                              method=method,
                              random_state=random_state,
                              vectorized=False)
    return bootstrap_res.confidence_interval.low, bootstrap_res.confidence_interval.high