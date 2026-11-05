"""Check the cycle-separation constant and the linear entropy proof.
Finite checks supplement, rather than replace, the compactness proof.
"""
import json
from pathlib import Path
import numpy as np


def cycle_mean(weights):
    n = len(weights)
    d = np.zeros((n+1, n))
    for k in range(1, n+1):
        d[k] = np.max(d[k-1, :, None] + weights, axis=0)
    return max(min((d[n,v]-d[k,v])/(n-k) for k in range(n)) for v in range(n))

rng = np.random.default_rng(20261010)
out = {}
for m in range(1,5):
    n = 2**m
    edges = [(w,y,(2*w+y)%n) for w in range(n) for y in (0,1)]
    minimum_lambda=1e100
    min_margin=1e100
    for _ in range(100):
        g=rng.uniform(-1,1,n); g/=max(abs(g))
        mat=np.full((n,n),-np.inf)
        for w,y,v in edges: mat[w,v]=(1-2*y)*g[w]
        lam=cycle_mean(mat); assert lam>0
        signs=1-2*np.array([[(w>>j)&1 for j in range(m)] for w in range(n)])
        for A in range(n):
            inds=[j for j in range(m) if A & (1<<j)]
            coeff=np.mean(g*np.prod(signs[:,inds],axis=1))
            span=max([j+1 for j in inds],default=0)
            assert lam>=abs(coeff)/(m+1-span)-1e-10
        minimum_lambda=min(minimum_lambda,lam)
        t=min(.25,lam/4)
        logmat=np.full_like(mat,-np.inf)
        mask=np.isfinite(mat); logmat[mask]=np.log1p(t*mat[mask])
        actual=cycle_mean(logmat)
        lower=lam*t-t*t/(2*(1-t))
        assert actual>=lower-1e-10
        min_margin=min(min_margin,actual-lower)
    out[m]={'minimum_lambda_sampled':minimum_lambda,'tested_directions':100,'minimum_margin':min_margin}
print(json.dumps(out,indent=2))
Path(__file__).with_name('cycle-linear-results.json').write_text(json.dumps(out,indent=2)+'\n')

sharp = {}
for m in range(1,7):
    n=2**m
    signs=1-2*np.array([[(w>>j)&1 for j in range(m)] for w in range(n)])
    chars=np.array([np.prod(signs[:,[j for j in range(m) if a&(1<<j)]],axis=1) for a in range(n)])
    G=sum(2**j*np.all(signs[:,:j]==-1,axis=1) for j in range(m+1))
    mat=np.full((n,n),-np.inf)
    for w in range(n):
        for y in (0,1):mat[w,(2*w+y)%n]=(1-2*y)*G[w]
    assert abs(cycle_mean(mat)-1)<1e-10
    assert abs(np.mean(G)-(m+1))<1e-10
    samples=[]
    for theta in (1e-4,1e-6):
        g=theta*G
        P=np.zeros((n,n)); logs=np.full((n,n),-np.inf)
        for w in range(n):
            for y in (0,1):
                v=(2*w+y)%n; P[w,v]=(1+(1-2*y)*g[w])/2
                logs[w,v]=np.log1p((1-2*y)*g[w])
        system=P.T-np.eye(n);system[-1]=1;rhs=np.zeros(n);rhs[-1]=1
        pi=np.linalg.solve(system,rhs)
        kappa=max(abs(chars@(pi*g)))
        loss=cycle_mean(logs)/np.log(2)
        ratio=loss/kappa
        target=1/((m+1)*np.log(2))
        assert abs(ratio/target-1)<100*theta*(2**m)
        samples.append({'theta':theta,'loss_over_kappa':ratio,'limit':target})
    sharp[m]=samples
print(json.dumps({'sharp_lower_construction':sharp},indent=2))
Path(__file__).with_name('cycle-linear-results.json').write_text(json.dumps({'cycle_checks':out,'sharp_lower_construction':sharp},indent=2)+'\n')
