"""Numerical boundary coverage for one implementation; no performance variants."""
MODES = ['native', 'avx512-disabled', 'hardware-disabled']
SHAPES = [(1, 1), (1, 8), (3, 7), (7, 9), (8, 31), (9, 32), (17, 33), (5, 63), (4, 64), (11, 65), (1024, 4096), (4096, 1024)]
FRAMES = [167, 89, 157, 83, 61, 151, 169, 48, 50, 225]


def cases(mode):
    assert mode in MODES
    rows = []
    if mode != 'hardware-disabled':
        for n, k in SHAPES:
            for special in [False, True]:
                rows.append(dict(id=f'row-{n}-{k}-' + ('special' if special else 'finite'), kind='row', n=n, k=k, special=special))
        for n, k in [(1024, 4096), (4096, 1024)]:
            for m in FRAMES:
                rows.append(dict(id=f'route-{n}-{k}-{m}', kind='route', n=n, k=k, m=m))
    for option in ['scalar', 'simd', 'auto']:
        rows.append(dict(id='fallback-' + option, kind='fallback', option=option))
    if mode == 'hardware-disabled':
        rows.append(dict(id='unavailable', kind='unavailable'))
    assert len(rows) == (4 if mode == 'hardware-disabled' else 47)
    assert len({r['id'] for r in rows}) == len(rows)
    return rows
