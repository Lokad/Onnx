# Uniform-mask Where candidate: normal AMD build

This four-job lane builds the isolated 423-input source without changing root.
The parent is qualified source81f75c38, Core672e5f30/Data065b7a7f. Require every
parent input and its normal-root V2 proof, plus closed Where capture bc0942f5.

Only Tensor<T>.Where may change: one float guard, one added local, one private
TryUniformScalarWhere helper. Compare all 3,189 existing Core and 697 Data
methods and flags, public surface and every original Where instruction/edge.
Only the declared local insertion and its index changes are permitted. Expect
3,190 candidate Core methods. The new helper is NoInlining|AggressiveOptimization;
all existing flags remain exact. Root product remains selected until complete
numerical, component, application, shared-model and package qualification.

Run `C:/Python313/python.exe -X utf8 -B run.py prepare`, then stage, launch,
observe, collect and `audit.py` separately. Freeze tools before staging. Tools
derive from the closed M55 build controller, with paths changed and the known
parent V2 release path selected before staging. Offline inputs/inspector/runtime
are reused. CPU2 computes, CPU0 monitors. Preflight12GiB available/3GiB tmpfs;
8GiB RSS,1GiB available/tmpfs minima,1GiB output/job,2GiB/campaign,900s/job,
four hours/campaign. Every owner must be terminal before any further VM work.

This lane proves build and compiled scope only, never semantics or performance.
Retain any failed worker or scope check and resolve the actual cause without
repeating unchanged performance trials.
