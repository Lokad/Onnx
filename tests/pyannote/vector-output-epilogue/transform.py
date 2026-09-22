"""Replace only the previously qualified scalar output loop."""
import difflib


def transform(source):
    begin = '        for (int channel = 0; channel < m; channel++)\n'
    assert source.count(begin) == 1
    start = source.index(begin); end = source.index('        return true;', start)
    old = source[start:end]
    assert 'AddBias(value, bias[channel])' in old and 'destination[channel * oh * ow + p] = value;' in old
    replacement = '        UnpackEpilogue(packedOutput, destination, bias, residual, m, oh * ow, lanes, relu);\n'
    result = source[:start]+replacement+source[end:]
    diff = ''.join(difflib.unified_diff(source.splitlines(True), result.splitlines(True), fromfile='qualified/BlockedSpatial.cs', tofile='vector-output/BlockedSpatial.cs'))
    return result, diff
