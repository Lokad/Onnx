# Parakeet projection observer: compiled scope

The diagnostic build passes with Core **f95a13c5 unchanged byte for byte**.
All eight AMD build jobs completed successfully. The isolated Data observer is
65cd9716; no model has yet run with it.

The original private Data Execute body is retained inside a disposable scope.
The other 696 existing Data methods and 161 existing consumer methods are
unchanged, including implementation flags and public interfaces. Consumer
changes initialize the observer, save metadata after the request clock and
check the selected assembly identities. Every original request/result check,
branch and exception region remains.

The original scope reviewer rejected nine compiler-generated methods for the
five-field JSON envelope. A separate correction checks exactly its Name, Pass,
Control, Frequency and Calls fields; method bodies use only the envelope and
standard equality/string/object helpers. The only external references create
and serialize it in the diagnostic Save method. A deliberately inserted product
call is rejected. All original compiled checks remain, and the original tools
and build are unchanged. No rebuild or inference retry occurred.

The observer reads actual prepared mappings, tensor layouts and the existing
scratch/copy accountants around all 217 constant projections. It also records
complete encoder node clocks. It does not invoke an arithmetic path or change
graph outputs/options. Counter bytes are requested sizes, not physical traffic;
source-predicted kernel names must remain distinct from instruction samples.

Next run the original full-corpus control and observed applications after
restoring the existing VM memory preflight. All native/public/ownership checks
remain required. Expected observations: 217 nodes times 80 requests = 17,360.
No performance admission or change to BENCHMARK.md follows from this build.

[Complete review](projection-build-20260924.json),
[exact correction](review_projection_build.py),
[frozen observation protocol](../projection-route-amd/README.md).

Build review SHA256: 74c37c401fb7213480bb9c09ef7f9a3124fd669552e328e4deab8d28abd7b4aa.
Archive: eaedbd8689d426e11d565fdd9dcc78360b01bc4d11828e201fc32d8a2ee8fa99.
Spec: 23f3b1699b34728f6c6789d507bf184022c4fcc9a6d39fc4d81e53cb332fbbd3.
