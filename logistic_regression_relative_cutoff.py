from sklearn.linear_model import LogisticRegression

import prediction


class MyLR(LogisticRegression):
    def __init__(
        self,
        # cutoff,
        relative_cutoff,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ) -> None:
        self.relative_cutoff = relative_cutoff
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio
        )


    def set_cutoff(self, cutoff):
        self.cutoff = cutoff

    def predict(self, X):
        pred_proba = self.predict_proba(X)[:, 1]
        class_pred, cutoff = prediction.predict_classification(pred_proba, self.relative_cutoff)
        return class_pred
