import sys, numpy as np, time
sys.path.insert(0, "/home/rasmitdevkota/projects/QAPTDRJMCMC")
import bayeswave_qclt as b
from multiprocessing import Pool
NI, NA = 80000, 10000
def job(a):
    m, s = a
    r = b.run(m, 100 + s, NI, NA, record_every=10)
    return m, s, r["K"], r["logL"]
if __name__ == "__main__":
    jobs = [(m, s) for m in b.METHODS for s in range(5)]
    t0 = time.time()
    with Pool(15) as p:
        res = p.map(job, jobs, chunksize=1)
    print(f"done in {time.time()-t0:.0f}s")
    for m in b.METHODS:
        K = np.stack([k for mm, s, k, l in res if mm == m]).astype(float)
        seg = K.shape[1] // 8
        print(f"{m:10s} <K> per 10k-iteration block: {np.round([K[:, i*seg:(i+1)*seg].mean() for i in range(8)], 2)}")
        print(f"{'':10s} per-seed <K> (last 40k): {np.round(K[:, K.shape[1]//2:].mean(1), 2)}")
