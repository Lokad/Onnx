"""Change dispatch and output storage only; retain arithmetic text exactly."""
import difflib
import re


def modify(source):
    changes = []

    def edit(name, transform):
        path = source / 'src/Lokad.Onnx' / name
        before = path.read_text(encoding='utf8')
        after = transform(before)
        assert before != after
        path.write_text(after, encoding='utf8')
        changes.append(dict(path='src/Lokad.Onnx/' + name, before=before, after=after))

    def append(text, method):
        pos = text.rindex('}')
        return text[:pos] + '\n' + method + '\n' + text[pos:]

    def provider(text, name):
        signature = re.search(r'    public static OpResult ' + name + r'\([^\n]+\)', text).group(0)
        internal = signature.replace('public static', 'internal static')[:-1] + ', TensorBufferPool? pool)'
        assert text.count(signature) == 1
        text = text.replace(signature, internal)
        if name == 'Conv':
            first = text.index('            case TensorElementType.Float:')
            last = text.index('            case TensorElementType.Double:', first)
            block = text[first:last]
            assert block.count('opts.Tensor)') == 2
            text = text[:first] + block.replace('opts.Tensor)', 'opts.Tensor, pool)') + text[last:]
        else:
            old = 'var inner = Conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides, options);'
            assert text.count(old) == 1
            text = text.replace(old, old[:-2] + ', pool);')
        wrapper = signature + ' =>\n        ' + name + '(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides, options, null);\n'
        return append(text, wrapper)

    edit('CPUExecutionProvider.ConvPool.cs', lambda t: provider(t, 'Conv'))
    edit('CPUExecutionProvider.Fusion.cs', lambda t: provider(t, 'ConvRelu'))

    def dispatch(text):
        suffix = 'Ints("strides"), opt),'
        for op in ['Conv', 'ConvRelu']:
            start = text.index('        OpType.' + op + ' =>')
            end = text.index('\n\n', start)
            block = text[start:end]
            assert block.count(suffix) == 1
            text = text[:start] + block.replace(suffix, 'Ints("strides"), opt, graph.ActivePool),') + text[end:]
        return text
    edit('Node.cs', dispatch)

    def tensor(text):
        matches = list(re.finditer(r'    public static Tensor<float> Conv2D\([^\n]+TensorExecutionOptions options\)', text))
        assert len(matches) == 2
        additions = []
        for match in reversed(matches):
            start = match.start()
            end = text.index('\n    }', match.end()) + len('\n    }')
            block = text[start:end]
            assert block.count('bias, options);') == 1
            extra = block.replace('public static', 'internal static', 1).replace('TensorExecutionOptions options)', 'TensorExecutionOptions options, TensorBufferPool? pool)', 1)
            additions.append(extra.replace('bias, options);', 'bias, options, pool);'))
            text = text[:start] + block.replace('bias, options);', 'bias, options, null);') + text[end:]
        signature = re.search(r'    static Tensor<float> Conv2DFloatCore\([^\n]+\)', text).group(0)
        text = text.replace(signature, signature[:-1] + ', TensorBufferPool? pool)', 1)
        old = '        var output = new DenseTensor<float>((ReadOnlySpan<int>)new int[] { N, M, outH, outW });'
        assert text.count(old) == 1
        new = '''        var dimensions = new int[] { N, M, outH, outW };
        // Preserve the fresh array's zero-initialization for every kernel path.
        var output = pool is null
            ? new DenseTensor<float>((ReadOnlySpan<int>)dimensions)
            : new DenseTensor<float>(new Memory<float>(pool.RentCleared<float>(checked(N * M * outH * outW))), dimensions);'''
        text = text.replace(old, new)
        for extra in reversed(additions):
            text = append(text, extra)
        return text
    edit('TensorOps.ConvPool.cs', tensor)
    return ''.join(''.join(difflib.unified_diff(c['before'].splitlines(True), c['after'].splitlines(True),
        fromfile='a/' + c['path'], tofile='b/' + c['path'])) for c in changes), changes
