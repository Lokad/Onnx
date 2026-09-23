# Method-specific correction for interleaved JIT output

The first M36 all-method capture remains failed, closure116194a1. In both
current-product widths, OutputWinograd's instrumented listing interleaved with
MultiplyWinograd's OSR listing. The four damaged parsed bodies are retained
verbatim. All348 numerical rows and172 resource samples passed independently;
both candidate traces were complete. No timing or product change occurred.

This successor runs exactly four serial diagnostic processes: current
MultiplyWinograd256, OutputWinograd256, MultiplyWinograd512, OutputWinograd512.
The consumer5619b769 and actual current Core521bae17 are copied unchanged from
closed numerical qualification47513c48. Each process executes all87 captured
calls and must reproduce its complete numerical reference rows and DLL hashes.
The JitDisasm filter is exactly one named private Core method per process;
the assembly goes to its own JitStdOutFile. Ordinary tiering, inlining, ISA
controls, consumer, fixtures and arithmetic remain unchanged.

Every emitted body must be complete, uninterleaved and for that exact method;
instrumented and full Tier1 bodies must both be present. Retain every OSR tier.
Do not reconstruct interleaved fragments or select favorable tiers. The later
review uses these complete streams for the four affected current methods,
retains all other complete original current methods and both full candidate
captures, and links the complete original failure. The original capture never
becomes passed. No timing screen is admitted before that combined review.

CPU2 workers, CPU0 monitor; SDK10.0.204/runtime10.0.8. Preserve12GiB available
and3GiB tmpfs preflight,8GiB owned RSS,1GiB available/tmpfs minimum and output
limit,2GiB total artifacts,900seconds per process. No build jobs or timing.
Use C:/Python313/python.exe -X utf8 -B with run.py prepare/stage/launch/observe/
collect, then audit.py. Only collect terminal identities. Refuse existing
artifacts/pyannote-winograd-output-blocks-codegen-amd-v2-20260923 and
/dev/shm/lokad-pyannote-winograd-output-blocks-codegen-v2-20260923.
