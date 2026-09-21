"""Keep existing partial-class method ordinals by appending additions last."""
import difflib
import re
from modify_source import modify as first_modify


def modify(source):
    _, changes = first_modify(source)
    before = {c['path']: c['before'] for c in changes}
    core = source / 'src/Lokad.Onnx'
    providers, tensors = [], []
    for filename, name in [('CPUExecutionProvider.ConvPool.cs', 'Conv'), ('CPUExecutionProvider.Fusion.cs', 'ConvRelu')]:
        path = core / filename
        text = path.read_text(encoding='utf8')
        matches = list(re.finditer(r'    public static OpResult ' + name + r'\([^\n]+\) =>\n[^\n]+;\n', text))
        assert len(matches) == 1
        match = matches[0]
        providers.append(match.group())
        path.write_text(text[:match.start()] + text[match.end():], encoding='utf8')
    path = core / 'TensorOps.ConvPool.cs'
    text = path.read_text(encoding='utf8')
    for _ in range(2):
        start = text.index('    internal static Tensor<float> Conv2D(')
        end = text.index('\n    }', start) + len('\n    }')
        tensors.append(text[start:end])
        text = text[:start] + text[end:]
    assert '    internal static Tensor<float> Conv2D(' not in text
    path.write_text(text, encoding='utf8')
    added = 'src/Lokad.Onnx/Zzz.ConvolutionPool.cs'
    text = '''namespace Lokad.Onnx;

using static Lokad.Onnx.MathOps;

// Compile new overloads after existing partial-class methods to keep the
// isolated candidate's unrelated compiler-generated method identities stable.
public partial class CPUExecutionProvider
{
''' + '\n'.join(providers) + '''}

public abstract partial class Tensor<T> where T : unmanaged
{
''' + '\n\n'.join(tensors) + '\n}\n'
    (source / added).write_text(text, encoding='utf8')
    before[added] = ''
    changes = [dict(path=name, before=old, after=(source / name).read_text(encoding='utf8')) for name, old in before.items()]
    patch = ''.join(''.join(difflib.unified_diff(c['before'].splitlines(True), c['after'].splitlines(True),
        fromfile='a/' + c['path'], tofile='b/' + c['path'])) for c in changes)
    return patch, changes
