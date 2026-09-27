"""Boundary contracts and every actual pointwise geometry, fixed before build."""
WIDTHS = [51,61,83,88,89,102,106,112,114,120,151,156,157,158,167,169,190,222,225]


def cases():
    rows=[]
    for m in [0,2,62,64,66,68,70,72]:
        for n in [0,1,63,64,65]:
            for rem in range(32):
                for exceptional in [False,True]:
                    rows.append(dict(m=m,n=n,k=32+rem,exceptional=exceptional,oracle=True))
    for m in [1024,2048]:
        for k in WIDTHS: rows.append(dict(m=m,n=1024,k=k,exceptional=False,oracle=False))
    assert len(rows)==2598
    return rows


FALLBACK = [[64,64,51],[66,65,89],[1024,64,61],[6,7,9]]
