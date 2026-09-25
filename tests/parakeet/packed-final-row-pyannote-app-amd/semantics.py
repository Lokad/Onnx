"""Only centroid floats may differ between products; own-product repeats remain exact."""
import math


def semantic(value):
    assert set(value) == {'status','windows','audio_seconds','intervals','exclusive_intervals','speakers'}
    for speaker in value['speakers']:
        assert set(speaker) == {'speaker','centroid','has_embedding'}
        assert len(speaker['centroid']) == 256
        assert all(type(x) in (int,float) and math.isfinite(x) for x in speaker['centroid'])
    return dict(value, speakers=[{k:v for k,v in speaker.items() if k != 'centroid'} for speaker in value['speakers']])


def compare_records(actual, expected, *, exact):
    assert len(actual) == len(expected)
    for a,e in zip(actual,expected,strict=True):
        assert (a['name'],a['phase'],a['pass']) == (e['name'],e['phase'],e['pass'])
        assert semantic(a['result']) == semantic(e['result']), 'Changed speaker timeline or status'
        if exact: assert a['result'] == e['result'], 'Changed own-product repeat'
    return dict(semantic_exact=True, full_results_exact=[r['result'] for r in actual] == [r['result'] for r in expected])
