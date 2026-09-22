"""Preserve the initial failure; change only bias selection for the actual caller."""
from pathlib import Path
import sys

TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS.parent / 'two-column'))
import prepare as prior


def configured():
    old=prior.BASE
    proof=prior.read(old / 'failure-closed.json');assert proof['passed'] and proof['expected_failure']
    prior.verify(proof['files'])
    for identity in proof['identities']:prior.terminal(identity)
    source=(TOOLS / 'prepare.py').read_text(encoding='utf8')
    changes=[('pyannote-direct-composition-20260922','pyannote-direct-composition-v2-20260922',1),
        ('from two_column_generator import generate','from span_bias_generator import generate, prior_generate',1),
        ("    assert kernel==(COMPONENT / 'normal/DirectOutput.cs').read_text(encoding='utf8')",
         "    assert prior_generate((source / 'src/Lokad.Onnx/MathOps.cs').read_text(encoding='utf-8-sig'))[0]==(COMPONENT / 'normal/DirectOutput.cs').read_text(encoding='utf8')",1),
        ("    shutil.copy2(TOOLS / 'ConvDirectOutput.cs',source / 'src/Lokad.Onnx/Zzz.ConvDirectOutput.cs')",
         "    wrapper=(TOOLS / 'ConvDirectOutput.cs').read_text(encoding='utf8')\n    assert wrapper.count('t[row * columns + col] + value')==1\n    wrapper=wrapper.replace('t[row * columns + col] + value','ConvDirectOutput.AddBiasScalar(t[row * columns + col], value)')\n    (source / 'src/Lokad.Onnx/Zzz.ConvDirectOutput.cs').write_text(wrapper,encoding='utf8')",1),
        ("            assert row['candidate_methods'][new_key]==value.replace('DirectOutput::','Lokad.Onnx.ConvDirectOutput::'),key",
         "            if key.split('::')[1] not in ['AddBiasScalar','AddBiasVector']:\n                assert row['candidate_methods'][new_key]==value.replace('DirectOutput::','Lokad.Onnx.ConvDirectOutput::'),key\n            else:\n                assert row['candidate_methods'][new_key]!=value.replace('DirectOutput::','Lokad.Onnx.ConvDirectOutput::'),key",1),
        ('compiled_kernel_bodies_equal=True','compiled_reduction_bodies_equal=True,actual_span_bias_successor=True',1)]
    for before,after,count in changes:
        assert source.count(before)==count,(before,source.count(before));source=source.replace(before,after)
    namespace=dict(__name__='direct_composition_successor',__file__=str(TOOLS / 'prepare.py'))
    exec(compile(source,str(TOOLS / 'prepare.py'),'exec'),namespace)
    return namespace


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','qualify']
    namespace=configured()
    if sys.argv[1]=='prepare':namespace['main']()
    else:
        source=(TOOLS / 'qualify.py').read_text(encoding='utf8')
        assert source.count('from prepare import *')==1
        source=source.replace('from prepare import *','')
        exec(compile(source,str(TOOLS / 'qualify.py'),'exec'),namespace)
        namespace['main']()
