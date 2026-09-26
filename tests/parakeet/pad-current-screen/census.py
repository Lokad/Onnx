"""Prospective synthetic Pad shapes using four previously recorded frame counts."""
def census():
    cases = []
    for frames in [51, 106, 167, 225]:
        cases.append(dict(name=f'attention-{frames}', shape=[1, 8, frames, 2*frames-1],
                          pads=[0, 0, 0, 1, 0, 0, 0, 0], fill=0, mode='constant', eligible=True))
        cases.append(dict(name=f'convolution-{frames}', shape=[1, 1024, frames],
                          pads=[0, 0, 4, 0, 0, 4], fill=0, mode='constant', eligible=True))
    for name, shape, pads, mode in [
        ('zero-padding', [1, 1024, 106], [0]*6, 'constant'),
        ('cropping', [1, 64, 225], [0, 0, -4, 0, 0, -4], 'constant'),
        ('outer-padding', [1, 64, 225], [0, 1, 0, 0, 2, 0], 'constant'),
        ('reflection', [1, 64, 225], [0, 0, 4, 0, 0, 4], 'reflect')]:
        cases.append(dict(name=name, shape=shape, pads=pads, mode=mode, fill=-7.5, eligible=False))
    return dict(passed=True, synthetic=True,
                provenance='Frame counts 51,106,167,225 are recorded; these constructed attention/convolution shapes are synthetic and not captured Pad intermediates. Pad widths are from all 48 actual encoder nodes.',
                cases=cases)
