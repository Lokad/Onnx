# Qualify the fixed eight-row column remainder

Build the single snapshot from pointwise-tail-source on the exclusive AMD VM.
Use qualified Core 47984318 and Data dd56902f as the baseline. Only one existing
Core method may differ; two private arithmetic helpers are added. Every other
Core/Data body, original implementation flag, public declaration and assembly
attribute must match. The application consumer and Data binary remain exact.

Raw checks invoke the original compiled kernel and candidate in the same process
through separate assembly load contexts. Each case accumulates three times into
nonzero destinations, compares every output bit, protects unaligned guard regions,
checks immutable A/B/packed buffers and requires zero warmed-call allocation.
The boundary set covers all 32 column remainders, rows 0/2/62/64/66/68/70/72,
reductions 0/1/63/64/65, finite and exceptional floats including distinct NaN
payloads. An independent per-element oracle checks these 2,560 cases. Another
38 checks use both actual row counts, reduction 1,024 and all 19 observed widths.

Run all 2,598 raw cases in normal and AVX512-disabled processes. Normal execution
also retains generated code for the old/new leaf and new helpers. This is a
numerical/code-generation run, never a timing score. Two hardware-disabled
processes run the exact same consumer with baseline and candidate binaries,
checking 12 public scalar/SIMD/automatic products each and explicit unsupported
intrinsics rejection. Keep every raw failure and finish the remaining contracts;
process completion is distinct from a passing candidate. Compare all successful
normal/AVX512-disabled output hashes and both scalar result sets.

The original VM worker uses CPU2 compute, CPU0 monitoring, half-second resource
observations and a 512 MiB output bound. Build preflight requires 2 GiB RAM /
1 GiB tmpfs, with a 3 GiB RSS / 180-second per-job cap. Contracts use the same
memory limits and at most 600 seconds per process. Leave 1 GiB RAM/tmpfs free.
Use SDK 10.0.204/runtime 10.0.8 and the retained offline feed. No downloads.

From the repository root, prefix each command with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/pointwise-tail-contracts-amd/run.py prepare
    tests/parakeet/pointwise-tail-contracts-amd/run.py stage
    tests/parakeet/pointwise-tail-contracts-amd/run.py launch build
    tests/parakeet/pointwise-tail-contracts-amd/run.py observe build
    tests/parakeet/pointwise-tail-contracts-amd/run.py collect build
    tests/parakeet/pointwise-tail-contracts-amd/audit.py build
    tests/parakeet/pointwise-tail-contracts-amd/run.py launch capture
    tests/parakeet/pointwise-tail-contracts-amd/run.py observe capture
    tests/parakeet/pointwise-tail-contracts-amd/run.py collect capture
    tests/parakeet/pointwise-tail-contracts-amd/audit.py capture

Prepare/stage/launch/collect/audit once, observe the same actual owner to terminal,
and preserve failures. The inspector receives its four required arguments.
After numerical closure, inspect generated code to confirm eight independent
accumulators, ordered FMA or separate multiply/add as applicable, and intact
full-panel arithmetic before measuring the predicted remainder-sensitive gain.
Passing contracts do not qualify the complete application or release.
