"""Educational SSM/PCA + fixed-PC logistic combination, not a published PDRP."""
from common import ROOT  # sets optional local dependency path
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class SSM(BaseEstimator, TransformerMixin):
    """Log -> remove each scan's mean -> remove TRAIN cohort voxel mean -> PCA."""

    def __init__(self, n_components=3):
        self.n_components = n_components

    @staticmethod
    def subject_center(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or not np.isfinite(X).all() or np.any(X <= 0):
            raise ValueError('Masked uptake must be finite, strictly positive, and 2D')
        logged = np.log(X)
        return logged - logged.mean(axis=1, keepdims=True)

    def fit(self, X, y=None):
        residual = self.subject_center(X)
        self.group_mean_ = residual.mean(axis=0)
        self.pca_ = PCA(n_components=self.n_components, svd_solver='randomized', random_state=42)
        self.pca_.fit(residual - self.group_mean_)
        return self

    def transform(self, X):
        return self.pca_.transform(self.subject_center(X) - self.group_mean_)


class PatternModel:
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X, y):
        y = np.asarray(y)
        if set(y) != {0, 1} or (y == 0).sum() < 2:
            raise ValueError('Need PD=1 and at least two training HC=0')
        self.ssm = SSM(self.n_components).fit(X)
        pc = self.ssm.transform(X)
        self.classifier = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000).fit(pc, y)
        self.pattern_ = self.classifier.coef_[0] @ self.ssm.pca_.components_
        hc = self.expression(X[y == 0])
        self.hc_mean_ = float(hc.mean())
        self.hc_std_ = float(hc.std(ddof=1))
        if self.hc_std_ <= 1e-12:
            raise ValueError('Training HC expression has zero variance')
        return self

    def expression(self, X):
        return self.ssm.transform(X) @ self.classifier.coef_[0]

    def score(self, X):
        return self.classifier.decision_function(self.ssm.transform(X))

    def zscore(self, X):
        return (self.expression(X) - self.hc_mean_) / self.hc_std_
