"""GPU backend for the pre-registered nested forward-selection curve in analyze_h1.py (same computation).

Per outer fold: greedy forward selection where each candidate set is scored by the sum over the 3 inner folds of the
variance-weighted R2 of RidgeCV(alphas=logspace(-3,3,7)) (sklearn GCV: efficient leave-one-out, intercept unpenalised,
one alpha for all targets chosen by the mean squared LOO error, first minimum). Ties: first best candidate (as the strict
'>' loop). Two exact shortcuts:
  (1) greedy selection is deterministic, so the path for k=30 contains the paths for k=5,10,15,20 as prefixes;
      it is computed once per fold up to the largest k;
  (2) all candidates of a step are scored in one batched SVD on the GPU (float64).
Equivalence with the sklearn loop is checked by test_equivalence()."""
import numpy as np, torch
from sklearn.model_selection import KFold
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = np.logspace(-3, 3, 7)


def _vw_r2_batch(Y, P):                       # Y (n,q), P (J,n,q) -> (J,)
    ss = ((Y - Y.mean(0)) ** 2).sum()
    return 1 - ((P - Y[None]) ** 2).sum((1, 2)) / ss


def _ridgecv_scores(Xtr, Ytr, Xte, Yte, chunk=64):
    """Xtr (J,n,m), Xte (J,t,m): RidgeCV-GCV fit on train, vw-R2 on test, per candidate."""
    out = []
    for s0 in range(0, Xtr.shape[0], chunk):
        A, B = Xtr[s0:s0 + chunk], Xte[s0:s0 + chunk]
        xm = A.mean(1, keepdim=True); ym = Ytr.mean(0, keepdim=True)
        Ac, Yc = A - xm, Ytr - ym; n = A.shape[1]
        U, s, Vh = torch.linalg.svd(Ac, full_matrices=False)          # (J,n,m), (J,m), (J,m,m)
        UtY = U.transpose(1, 2) @ Yc                                    # (J,m,q)
        s2 = s ** 2; best_mse = torch.full((A.shape[0],), float("inf"), device=DEV, dtype=A.dtype)
        best_a = torch.zeros(A.shape[0], device=DEV, dtype=A.dtype)
        for a in ALPHAS:
            f = s2 / (s2 + a)
            yhat = U @ (f[:, :, None] * UtY)
            h = 1.0 / n + (U ** 2 * f[:, None, :]).sum(2)
            mse = (((Yc[None] - yhat) / (1 - h)[:, :, None]) ** 2).mean((1, 2))
            upd = mse < best_mse; best_mse = torch.where(upd, mse, best_mse); best_a = torch.where(upd, torch.full_like(best_a, a), best_a)
        coef = Vh.transpose(1, 2) @ ((s / (s2 + best_a[:, None]))[:, :, None] * UtY)   # (J,m,q)
        P = (B - xm) @ coef + ym[None]
        out.append(_vw_r2_batch(Yte, P))
    return torch.cat(out)


def greedy_path(Xtr, Ytr, kmax, seed):
    """Selection order up to kmax for one outer training fold (Xtr already imputed and standardised)."""
    X = torch.as_tensor(Xtr, dtype=torch.float64, device=DEV); Y = torch.as_tensor(Ytr, dtype=torch.float64, device=DEV)
    inner = [(torch.as_tensor(a, device=DEV), torch.as_tensor(b, device=DEV)) for a, b in KFold(3, shuffle=True, random_state=seed).split(Xtr)]
    chosen = []
    while len(chosen) < kmax:
        cand = [j for j in range(X.shape[1]) if j not in chosen]
        cols = torch.as_tensor([chosen + [j] for j in cand], device=DEV)            # (J,m)
        tot = torch.zeros(len(cand), dtype=torch.float64, device=DEV)
        for itr, ite in inner:
            Xa = X[itr][:, cols].permute(1, 0, 2); Xb = X[ite][:, cols].permute(1, 0, 2)  # (J,n,m)
            tot += _ridgecv_scores(Xa, Y[itr], Xb, Y[ite])
        chosen.append(cand[int(torch.argmax(tot))])                                 # first maximum
    return chosen


def test_equivalence(n=240, p=12, q=8, k=4, seed=0):
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(1); X = rng.normal(size=(n, p)); Y = X[:, :5] @ rng.normal(size=(5, q)) + rng.normal(size=(n, q))
    inner = list(KFold(3, shuffle=True, random_state=seed).split(X)); chosen = []
    while len(chosen) < k:
        best, bs = None, -np.inf
        for j in range(p):
            if j in chosen: continue
            cols = chosen + [j]; sc = 0.0
            for itr, ite in inner:
                m = RidgeCV(alphas=ALPHAS).fit(X[itr][:, cols], Y[itr]); sc += r2_score(Y[ite], m.predict(X[ite][:, cols]), multioutput="variance_weighted")
            if sc > bs: best, bs = j, sc
        chosen.append(best)
    return {"sklearn_path": chosen, "gpu_path": greedy_path(X, Y, k, seed), "device": DEV}


if __name__ == "__main__":
    import json; print(json.dumps(test_equivalence()))
