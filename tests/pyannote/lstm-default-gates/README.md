# Default LSTM gate specialization: isolated experiment

M25's wider projections missed the fixed complete-call gate. This distinct
candidate retains the current product's projections and specializes only
the default sigmoid/tanh/tanh state update with bias, no clipping/peepholes
and uncoupled input/forget. The generic loop remains byte-for-byte intact.
All MathF operations, addition order, mutable-bias behavior, execution-mode
controls, scratch and output ownership remain mandatory.

The current AMD Lstm Tier1-OSR body is9,873bytes and contains five indirect
activation sites. The pinned ORT2e2543fbe9fae542f921d47a72d21d5a4ef0b710
uni_directional_lstm.cc instead invokes activations on hidden-size arrays;
rnn_helpers.cc uses MlasComputeLogistic/MlasComputeTanh. This source comparison
suggests a dispatch specialization. It neither measures the indirect-call
cost nor proves the installed ORT kernel. No ORT approximation or bias fusion
is copied into this candidate.

Run C:/Python313/python.exe -X utf8 -B
tests/pyannote/lstm-default-gates/prepare.py from the repository root.
It creates artifacts/pyannote-lstm-default-gates-20260923, preserves all416
qualified root source hashes and emits the exact isolated patch and source
receipt. New complete-operator tests force the retained generic loop with
positive-infinite clipping as a finite-input differential reference. Cover
eight sequence lengths, all directions, explicit/implicit defaults, execution
modes, initial states, valid-prefix lengths, bias mutation and held outputs.

Preparation does not build or qualify the candidate. Next use a fresh AMD
build namespace with SDK10.0.204/runtime10.0.8 and global.json. Require only
Lstm to change and one private helper to be added in the compiled inventory;
all other3162Core/697Data methods and public declarations remain identical.
Then focused/complete captured-model numerics, generated-code inspection and
the unchanged10%complete-LSTM screen precede any application work. Full
application/native/package/long-meeting qualification precedes integration.
Pyannote remains first, Parakeet second, Whisper deferred.
