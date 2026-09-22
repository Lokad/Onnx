"""Replace only scalar input copies after all original checks and padding clear."""
import difflib


def transform(source):
    start = source.index('        for (int channel = 0; channel < c; channel++)')
    end = source.index('    }', start)
    old = source[start:end]
    assert 'channel / lanes * (h + 2)' in old and '= input[(channel * h + y) * w + x];' in old
    replacement = '        PackInputTiles(input, packed, c, h, w, lanes);\n'
    result = source[:start]+replacement+source[end:]
    diff = ''.join(difflib.unified_diff(source.splitlines(True), result.splitlines(True), fromfile='qualified/BlockedSpatial.cs', tofile='vector-input/BlockedSpatial.cs'))
    return result, diff
