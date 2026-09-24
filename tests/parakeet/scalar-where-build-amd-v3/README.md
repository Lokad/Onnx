# Uniform-mask Where candidate: normal AMD build

This four-job lane builds the isolated 423-input source without changing root.
The parent is qualified source81f75c38, Core672e5f30/Data065b7a7f. Require every
parent input and its normal-root V2 proof, plus closed Where capture bc0942f5.

Only Tensor<T>.Where may change: one float guard, one added local, one internal
UniformScalarWhere.Try<T> helper in a separate class. Compare all 3,189 existing Core and 697 Data
methods and flags, public surface and every original Where instruction/edge.
Only the declared local insertion and its index changes are permitted. Expect
3,190 candidate Core methods. The new helper is NoInlining|AggressiveOptimization;
all existing flags remain exact. Root product remains selected until complete
numerical, component, application, shared-model and package qualification.

Run `C:/Python313/python.exe -X utf8 -B run.py prepare`, then stage, launch,
observe, collect and `audit.py` separately. Freeze tools before staging. Tools
derive from the frozen M56 V1 build controller, with paths changed and the known
parent V2 release path selected before staging. Offline inputs/inspector/runtime
are reused. CPU2 computes, CPU0 monitors. Preflight12GiB available/3GiB tmpfs;
8GiB RSS,1GiB available/tmpfs minima,1GiB output/job,2GiB/campaign,900s/job,
four hours/campaign. Every owner must be terminal before any further VM work.

This lane proves build and compiled scope only, never semantics or performance.
Retain any failed worker or scope check and resolve the actual cause without
repeating unchanged performance trials.

V1 compiled, but failed scope because a new Tensor method renumbered existing
compiler-generated matrix closures. V2 moves the unchanged helper algorithm to
a separate internal class. No candidate timing has run; retain V1 failed proof
faa41f5e. Require the original strict no-renaming scope in this lane.

V3 preserves the separate helper class and every original scope constraint.
The eight-case diagnostic e83bf8c6 proves V2 differs from selected Where on
raw masks[0,2]and[0,255]. V3 tests zero/nonzero bytes with IndexOf(0) or
IndexOfAnyExcept(0), preserving existing public tensor truth semantics. The
123-case canonical census and eight additional raw masks must qualify anew.
No M56 candidate timing has run.
