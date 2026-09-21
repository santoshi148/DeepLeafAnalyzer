import numpy as np

def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    x=np.asarray(values,float); rng=np.random.default_rng(seed)
    boots=np.array([rng.choice(x,size=len(x),replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(x.std(ddof=1) if len(x)>1 else 0.0), tuple(np.quantile(boots,[alpha/2,1-alpha/2]))
