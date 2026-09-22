# Inspect four-block convolution on AMD

This diagnostic loads the exact selected and candidate Core and LayerGraphs
assemblies, without rebuilding them. One new outer driver binds the existing
fixture constructor and ordinary graph callers by reflection. Four fixed passes
over all 108 captured calls require 432 exact output hashes in each process.
Readonly operands and original graph-dispatch assertions remain enforced.

The only runtime flag is `DOTNET_JitDisasm=Kernel512*`. Instruction-set and tiering
settings stay ordinary. Local AVX2 runs check both driver/assembly combinations;
two fresh AMD AVX512 processes capture every emitted compilation tier. This is a
separate diagnostic, whose JIT profiling state is not assumed identical to timing.

Use `C:/Python313/python.exe -X utf8 -B` from root:

    tests/pyannote/input-address-codegen/run.py prepare
    tests/pyannote/input-address-codegen/run.py stage
    tests/pyannote/input-address-codegen/run.py launch
    tests/pyannote/input-address-codegen/run.py observe
    tests/pyannote/input-address-codegen/run.py collect
    tests/pyannote/input-address-codegen/audit.py

All existing numerical-lane resource bounds remain. Collection requires terminal
owners. The audit requires complete optimized Tier1 listings for selected
Kernel512 in both roles, retaining raw logs and every
tier. It reports code bytes, reduction blocks, broadcast instructions and vector
stack references. None of these counts establishes a performance ratio or admits
the candidate; a separate fixed complete-call comparison remains required.
