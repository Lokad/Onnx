# Build the isolated prepared decoder recurrence

This prospective build binds source `b52a89c1`, actual capture `28c7afe4`, selected
Core `672e5f30` / Data `065b7a7f`, and root qualification `16d57081`. It builds the
CLI and its product dependencies from the isolated 424-file snapshot. No root
product is edited and no inference or timing occurs in this lane.

Four jobs run on AMD CPU2: SDK version, CLI restore, CLI build and the retained
compiled-method inventory. The CPU0 monitor records every thread, owned PID/birth,
resource sample and immutable input. Canonical global.json selects SDK10.0.204
and runtime10.0.8; all .NET commands use `--tl:off --nologo -v minimal`. Reuse the
existing offline feed, selected runtime and inventory helper; no download.

This build-only namespace has prospective bounds: 8 GiB available memory and
2 GiB free tmpfs at preflight; RSS below 8 GiB, at least 1 GiB available memory and
tmpfs throughout; 1 GiB per-job outputs, 2 GiB campaign artifacts, 900 seconds/job
and four hours total. These bounds concern a source build with no model load;
full-model/application bounds are unchanged. Namespace:
`/dev/shm/lokad-parakeet-prepared-recurrence-build-20260924`.

The inventory must preserve the public interface, every existing method flag,
all 697 Data methods, and every Core method outside the explicitly changed
provider dispatch, graph constructor/lifecycle/context, packing refresh and
option-record initialization/equality/hash methods. The option record's typed
equality can change with its new internal field; its object wrapper cannot.
New methods belong only to the internal `GraphLstmPacking`, `PackedLstmWeight`,
their compiler-generated nested types, and the internal option property's two
accessors. Every new method has the ordinary implementation flags. The existing
ordered projection and panel kernels remain byte-exact; the LSTM body must call
the prepared resolver twice and the unchanged projection twice. The frozen
source patch separately limits provider edits to the route/dispatch additions.
This build review does not substitute for numerical or lifecycle tests.

From repository root use `C:/Python313/python.exe -X utf8 -B` with this directory's
`run.py prepare`, `stage`, `launch`, `observe` while active, then `collect` after
all owners terminate. Run `audit.py` to independently reconcile compiled scope
and every resource observation. Keep failures, use named successors for harness
fixes, never overwrite hardlinked products and never observe a closed campaign.

Next qualify the 22 new focused contracts plus retained LSTM/packing contracts in
both instruction modes, actual decoder residency/dispatch, full captured/native/
public results, then frozen complete-call and application comparisons. No release
or BENCHMARK change follows from compilation alone.
