# Resolve the remaining sigmoid cost before changing Parakeet

ORT's actual alpha-1 QuickGelu caller and AVX-512 SiLU loop are already proved
and bound to this workload. Reuse that proof. The current 72 managed activations
cost 0.852870 seconds in Sigmoid and 0.115477 in multiplication, compared with
ORT's 0.174952 seconds for both. Determine whether the remaining managed sigmoid
cost is generated arithmetic or allocation/collection/surrounding execution.

Run one control and one sampled process of the unchanged twenty-clip application,
one warmup and three measured passes each. Reuse qualified Core e98edee2, Data
7f4dd050 and consumer 38ab5c7e from the completed transpose profile. No rebuild,
product edit, ORT inference, approximation trial or benchmark score belongs here.

Both processes enable exactly two logging options: `DOTNET_JitDisasm` selects
`CPUExecutionProvider:Sigmoid` and `CPUExecutionProvider:SigmoidRationalVector`;
`DOTNET_JitDisasmWithCodeBytes=1`. All other runtime/product options stay clean.
The diagnostic adapter first asserts this exact dictionary, then passes a copy
with empty flags to the original application validator. Raw results retain the
flags. No original validator or release acceptance rule is changed.

The retained EventPipe collector attaches after all twenty warmups and before
the sixty measured requests. It runs on CPU0; inference runs on CPU2. Providers
are runtime GC/loader/JIT (`0x1019:5`), sampled thread stacks (`0x0:5`) and all
keywords of the existing request-boundary provider. Preserve raw nettrace,
Speedscope and Chromium stacks, complete event records, lost-event counts,
complete public results and per-request CPU/allocation/GC counters. Standard
stack exports identify methods, not instruction addresses or a sampled JIT tier.
Disassembly alone does not prove which tier ran at each sample. Unresolved
native frames stay unresolved; GC pauses are application observations unless
their association with an operator is proved.

Reuse the retained request-pair worker and existing tracer/exporter without
building either. Commands, from the repository root with Python 3.13:

    python -X utf8 -B tests/parakeet/sigmoid-residual-diagnostic-amd/run.py prepare
    python -X utf8 -B tests/parakeet/sigmoid-residual-diagnostic-amd/run.py stage
    python -X utf8 -B tests/parakeet/sigmoid-residual-diagnostic-amd/run.py launch
    python -X utf8 -B tests/parakeet/sigmoid-residual-diagnostic-amd/run.py observe
    python -X utf8 -B tests/parakeet/sigmoid-residual-diagnostic-amd/run.py collect
    python -X utf8 -B tests/parakeet/sigmoid-residual-diagnostic-amd/audit.py

Except observation, execute each lifecycle step once. Preserve any failure before
choosing a repair. Before launching, require verified inputs, idle VM, the same
boot, 11 GiB available and 2 GiB tmpfs. Bound each process at 900 seconds, owned
RSS at 12 GiB, remaining memory/tmpfs at 1 GiB, and all output at 512 MiB. Local
storage is counted by unique allocated bytes; refresh it after collection.

Require all 160 request results and ownership checks; exact qualified outputs;
zero lost events; all 120 measured request boundary events in order; matching
stack exports; and sampled full-request duration within 5% of measured request
wall time. Report overhead against the matched logging control and separately
against the previous uninstrumented control, without subtraction. A diagnostic
failure does not justify relaxing a requirement or rerunning unchanged work.

Publish the measured distinction and its limitations, then choose at most one
cause and one predicted intervention. A later product candidate still needs
the original independent complete-application gates and release qualification.
