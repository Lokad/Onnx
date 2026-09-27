"""Run the existing backend tests in one explicitly recorded hardware mode."""
import os
from pathlib import Path
import sys

mode, base, output = sys.argv[1:4]
assert mode in ['normal', 'noavx512', 'scalar'] and sys.argv[4] == '/home/vermorel/.dotnet/dotnet'
env = dict(os.environ)
assert not any(k.lower().startswith(('dotnet_enable', 'complus_', 'lokad_')) for k in env)
if mode != 'normal': env['DOTNET_EnableAVX512' if mode == 'noavx512' else 'DOTNET_EnableHWIntrinsic'] = '0'
env.update(LSTM_LAYOUT_BASE=base, LSTM_LAYOUT_MODE=mode,
    LSTM_LAYOUT_IDENTITY=str(Path(output)/'loaded.json'),
    LSTM_LAYOUT_PROJECTIONS=str(Path(output)/'projections.json'))
os.execve(sys.argv[4], sys.argv[4:], env)
