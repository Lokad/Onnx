"""Split only the AVX512 reduction traversal; preserve each output's operation order."""
import difflib
import hashlib

SOURCE='src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs'
SELECTED='b7f2730aca76bdeea61bf6ad168c9b9713e4dce942045725dfa34485bd4187cf'

def transform(text):
    original=text.replace('\r\n','\n')
    assert hashlib.sha256(original.encode()).hexdigest()==SELECTED
    start=original.index('    static void Kernel512(')
    prefix,kernel=original[:start],original[start:]
    old=('        for (int y = 0; y < oh; y++)\n'
         '        {\n'
         '            int col = 0;')
    new=('        for (int y = 0; y < oh; y++)\n'
         '        for (int firstChannel = 0; firstChannel < c; firstChannel += c >= 64 ? 16 : c)\n'
         '        {\n'
         '            int channelEnd = firstChannel + (c >= 64 ? 16 : c);\n'
         '            int col = 0;')
    assert kernel.count(old)==1;kernel=kernel.replace(old,new)
    for position in range(6):
        for block in range(2):
            old=f'                Vector512<float> a{position}{block} = Vector512<float>.Zero;'
            offset=f'(oc / lanes{(" + 1") if block else ""}) * spatial + y * ow + col + {position}'
            condition='firstChannel == 0'+(' || oc + lanes >= m' if block else '')
            new=(f'                Vector512<float> a{position}{block} = {condition}\n'
                 f'                    ? Vector512<float>.Zero : *(Vector512<float>*)(output + ({offset}) * lanes);')
            assert kernel.count(old)==1;kernel=kernel.replace(old,new)
    old='                var a0 = Vector512<float>.Zero; var a1 = Vector512<float>.Zero;'
    new=('                var a0 = firstChannel == 0 ? Vector512<float>.Zero\n'
         '                    : *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col) * lanes);\n'
         '                var a1 = firstChannel == 0 || oc + lanes >= m ? Vector512<float>.Zero\n'
         '                    : *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col) * lanes);')
    assert kernel.count(old)==1;kernel=kernel.replace(old,new)
    old='weights + oc * c * 9';assert kernel.count(old)==2
    kernel=kernel.replace(old,old+' + firstChannel * 9 * lanes')
    old='for (int ic = 0; ic < c; ic++)';assert kernel.count(old)==2
    kernel=kernel.replace(old,'for (int ic = firstChannel; ic < channelEnd; ic++)')
    candidate=prefix+kernel
    assert candidate[:start]==prefix
    # Every reduction operation and store remains byte-exact in the same order.
    def operations(value):
        return [line for line in value.splitlines() if any(token in line for token in
            ['FusedMultiplyAdd(', 'a0 = a0 + input', 'w0 += lanes;', '*(Vector512<float>*)(output']) and not '? Vector512' in line and not line.lstrip().startswith(':')]
    assert operations(kernel)==operations(original[start:])
    difference=''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='a/'+SOURCE,tofile='b/'+SOURCE))
    return candidate,difference
