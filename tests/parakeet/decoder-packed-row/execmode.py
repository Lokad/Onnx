"""Replace this process with a correctness consumer in one declared hardware mode."""
import os
import sys

assert sys.argv[1] in ['noavx512', 'scalar'] and sys.argv[2] == '/home/vermorel/.dotnet/dotnet'
key = 'DOTNET_EnableAVX512' if sys.argv[1] == 'noavx512' else 'DOTNET_EnableHWIntrinsic'
env = dict(os.environ)
assert not any(k.lower().startswith(('dotnet_', 'complus_', 'lokad_')) for k in env)
env[key] = '0'
os.execve(sys.argv[2], sys.argv[2:], env)
