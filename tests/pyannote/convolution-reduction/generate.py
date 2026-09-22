"""Generate only the prospective full-panel loop transformation, not a product patch."""
from pathlib import Path


def generate(source):
    signature = 'public unsafe static void mm_unsafe_vectorized_intrinsics_3x4packed('
    assert source.count(signature) == 1
    start = source.index(signature)
    brace = source.index('{', start)
    depth = 1; end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}'); end += 1
    method = source[start:end]
    row_loop = 'for (int i = 0; i < M; i += 3)'
    assert method.count(row_loop) == 3
    opening = method.index(row_loop)
    row_brace = method.index('{', opening)
    depth = 1; closing = row_brace + 1
    while depth:
        depth += (method[closing] == '{') - (method[closing] == '}'); closing += 1
    row = method[opening:closing]
    reduction = 'for (int j = 0; j < N; ++j)'
    assert row.count(reduction) == 1 and method.count(reduction) == 4
    updated = row.replace(reduction, 'for (int j = reductionStart; j < reductionEnd; ++j)')
    updated = ('for (int reductionStart = 0, reductionEnd; reductionStart < N; reductionStart = reductionEnd)\n'
        '            {\n'
        '                reductionEnd = reductionStart + Math.Min(128, N - reductionStart);\n'
        '                '+updated+'\n            }')
    modified = method[:opening] + updated + method[closing:]
    modified = modified.replace(signature, 'public unsafe static void Multiply(')
    assert modified[modified.index('int rem = K - blocked;'):] == method[method.index('int rem = K - blocked;'):]
    header = ('// Generated from the selected three-row kernel. Only the full-panel row\n'
        '// loop is wrapped in 128-term reduction blocks; tail code is identical.\n'
        'using System;\nusing System.Runtime.CompilerServices;\nusing System.Runtime.Intrinsics;\n'
        'using System.Runtime.Intrinsics.X86;\n\nstatic class Block128\n{\n    ')
    return header + modified + '\n}\n', method
