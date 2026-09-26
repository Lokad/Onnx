"""Validate all priming calls and the prospectively fixed elapsed stopping rule."""


def validate(prefix, suffix):
    assert prefix['passed'] and prefix['protocol'] == 'pad-census-ten-seconds-after-first-v1'
    assert prefix['pid'] == suffix['pid'] and prefix['frequency'] == suffix['frequency'] > 0
    frequency = prefix['frequency']; rounds = prefix['passes']
    assert 2 <= len(rounds) <= 16 and prefix['calls'] == len(rounds) * 9360
    assert prefix['firstEnd'] == rounds[0]['end'] and prefix['ended'] == rounds[-1]['end']
    assert prefix['began'] < rounds[0]['start']
    assert 0 < prefix['ended'] - prefix['began'] < 180 * frequency
    assert prefix['ended'] - prefix['firstEnd'] >= 10 * frequency
    assert rounds[-2]['end'] - prefix['firstEnd'] < 10 * frequency
    assert prefix['ended'] < suffix['suffixStart']
    prior = prefix['began']
    for index, group in enumerate(rounds):
        assert group['round'] == index and prior < group['start']
        assert len(group['rows']) == len(suffix['rows']) == 12
        prior = group['start']
        for row, expected in zip(group['rows'], suffix['rows'], strict=True):
            for key in ('index', 'name', 'shape', 'pads', 'mode', 'fill', 'output', 'exact', 'inputs', 'ownership'):
                assert row[key] == expected[key], key
            assert type(row['setupTicks']) is int and row['setupTicks'] > 0
            assert len(row['clocks']) == 780
            for iteration, clock in enumerate(row['clocks']):
                assert type(clock['iteration']) is int and clock['iteration'] == iteration
                assert type(clock['warmup']) is bool and clock['warmup'] == (iteration < 600)
                assert all(type(clock[key]) is int for key in ('start', 'stop', 'ticks'))
                assert prior < clock['start'] < clock['stop'] and clock['ticks'] == clock['stop'] - clock['start']
                prior = clock['stop']
        assert prior < group['end']; prior = group['end']
    return dict(passed=True, rounds=len(rounds), calls=prefix['calls'],
        seconds=(prefix['ended'] - prefix['began']) / frequency,
        seconds_after_first=(prefix['ended'] - prefix['firstEnd']) / frequency)
