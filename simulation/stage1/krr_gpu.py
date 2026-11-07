"""GPU backend for the pre-registered RBF kernel ridge grid search (analyze_h1.krr), same estimator, float64.

Reproduces sklearn GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": A, "gamma": G}, cv=KFold(5, shuffle=True,
random_state=SEED), scoring=variance-weighted R2) followed by refit on the whole training set:
  - candidate order = sklearn ParameterGrid order (alpha outer, gamma inner); best = first candidate with the highest
    mean (unweighted over folds) inner R2, as GridSearchCV's rank_test_score.argmin();
  - KernelRidge: dual = (K + alpha I)^-1 Y with K = exp(-gamma ||x - x'||^2); here solved through one eigendecomposition
    of K per (fold, gamma), which is the same linear solve for every alpha.
Equivalence with sklearn is checked by test_equivalence() (best params identical, predictions to ~1e-8)."""
import numpy as np, torch
from sklearn.model_selection import KFold

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _sqd(A, B):
    return ((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T).clamp_min(0)


def _vw_r2(Y, P):
    ss = ((Y - Y.mean(0)) ** 2).sum(0); return float(1 - ((Y - P) ** 2).sum() / ss.sum())


class GPUKRRGrid:
    def __init__(self, alphas, gammas, seed, inner=5):
        self.alphas, self.gammas, self.seed, self.inner = list(alphas), list(gammas), seed, inner

    def fit(self, X, Y):
        X = torch.as_tensor(np.asarray(X, np.float64), device=DEV); Yt = torch.as_tensor(np.asarray(Y, np.float64), device=DEV)
        cand = [(a, g) for a in self.alphas for g in self.gammas]; sc = np.zeros((len(cand), self.inner))
        for f, (tr, te) in enumerate(KFold(self.inner, shuffle=True, random_state=self.seed).split(np.zeros(len(X)))):
            tr, te = torch.as_tensor(tr, device=DEV), torch.as_tensor(te, device=DEV)
            Dtr, Dte = _sqd(X[tr], X[tr]), _sqd(X[te], X[tr])
            for gi, g in enumerate(self.gammas):
                lam, Q = torch.linalg.eigh(torch.exp(-g * Dtr)); QtY = Q.T @ Yt[tr]; Kte = torch.exp(-g * Dte)
                for ai, a in enumerate(self.alphas):
                    P = Kte @ (Q @ (QtY / (lam + a)[:, None]))
                    sc[ai * len(self.gammas) + gi, f] = _vw_r2(Yt[te], P)
        self.cv_scores_ = sc; best = int(np.argmax(sc.mean(1)))       # first maximum, as rank_test_score.argmin()
        a, g = cand[best]; self.best_params_ = {"alpha": a, "gamma": g}
        K = torch.exp(-g * _sqd(X, X)); K.diagonal().add_(a)
        self.dual_ = torch.linalg.solve(K, Yt); self.X_ = X; self.g_ = g
        return self

    def predict(self, Xn):
        Xn = torch.as_tensor(np.asarray(Xn, np.float64), device=DEV)
        return (torch.exp(-self.g_ * _sqd(Xn, self.X_)) @ self.dual_).cpu().numpy()


def test_equivalence(n=500, p=40, q=64, seed=3):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, r2_score
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, p)); W = rng.normal(size=(p, q))
    Y = np.tanh(X @ W / 5) + 0.3 * rng.normal(size=(n, q)); Xt = rng.normal(size=(100, p))
    grid = {"alpha": [1e-3, 1e-2, 1e-1, 1], "gamma": [0.005, 0.01, 0.02, 0.05, 0.1]}
    sk = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=KFold(5, shuffle=True, random_state=0),
                      scoring=make_scorer(r2_score, multioutput="variance_weighted")).fit(X, Y)
    gp = GPUKRRGrid(grid["alpha"], grid["gamma"], 0).fit(X, Y)
    return {"sklearn_best": sk.best_params_, "gpu_best": gp.best_params_,
            "max_abs_cv_score_diff": float(np.abs(sk.cv_results_["mean_test_score"] - gp.cv_scores_.mean(1)).max()),
            "max_abs_pred_diff": float(np.abs(sk.predict(Xt) - gp.predict(Xt)).max()), "device": DEV}


if __name__ == "__main__":
    import json; print(json.dumps(test_equivalence(), indent=1))
