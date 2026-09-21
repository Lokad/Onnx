# Corrected private memory candidate: source policy and package checks

The [first coherent build](../memory-product/failure-20260921.md) failed two
repository source-policy tests. This distinct attempt replaces the four Data
null-forgiving operators with an accurate `NotNullWhen(true)` annotation and
replaces optional test parameters with explicit overloads. Typed test placeholders
also remove four nullable warnings. Original source and failures remain intact.

`prepare.py` copies the verified prior source and changes only those two files.
`run.py --artifact artifacts/whisper-memory-product-v2-20260921` builds all eight
projects, runs both complete test suites, packs the new core and runs four private
package consumers. `audit.py` with the same argument checks every saved result,
budget/ownership contract, package/source identity and terminal process birth.
Use `C:/Python313/python.exe -X utf8 -B` from the repository root.

No model inference, VM workload, production promotion or package publication occurs
in this lane. The pre-correction audio qualification remains separate evidence;
the compiled runtime effect of the nullable annotation must be checked before
using that evidence for the corrected source.
