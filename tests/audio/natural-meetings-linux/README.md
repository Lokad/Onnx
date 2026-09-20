# Native Whisper reference on Linux

This continuation completes the natural-meeting native reference on the exclusive
AMD VM. The original Windows Whisper worker stopped at its available-memory guard
after completing the first meeting. A declared Windows retry then refused after
fifteen minutes of unstable launch headroom, without creating an inference worker.
Both attempts remain evidence; the already completed AMD managed calls and
Parakeet native calls are not repeated.

The new `artifacts/asr-native-linux-20260920` directory holds an isolated CPU-only
Torch 2.11.0+cpu installation, with official wheel download/hash reports. Missing
dependencies use SymPy 1.13.3, NetworkX 3.6.1 and mpmath 1.3.0. Existing NumPy
2.2.4, Microsoft ORT 1.29.0, Transformers 5.16.1 and tokenizers 0.23.2 remain
unchanged. Original model and audio files are reused by hash.

`dependencies.py --root <checkout> --artifact <new-artifact> --original
<original-natural-ASR-artifact>` checks imports, CPU/thread settings, shared-library
identities and the original frontend/decoder source. It performs no neural
inference and refuses an existing output. Python 3.12 and 3.13 format `ast.dump`
differently; a recursively serialized tree of every node type, field and value
proves that both borrowed decoder function bodies are identical. The portable
body hash is `43272a768eadeac0fd6eba53bee1c79ec6cf5f86aab3db8af8b6e7cab34893d2`.
Original generator and frontend file hashes are checked independently.

The Linux runner is prepared, its input-only proof passes, and the frozen recording
sequence is running. Source is `4a768fd`; frozen manifest SHA256 is
`9f866075940948a4d0d776188675fe255f10e2f2377e385afa7712462c1bc514`.
The schedule remains ES2004a 600 seconds, IS1009a 600 seconds, and ES2004a
30-second recovery in one fresh native process, with unchanged recording options
and explicit English. The VM profile requires 13 GiB available at launch, less
than 14 GiB group RSS, at least 1 GiB available during execution, and at most two
hours. Recording results and final comparison remain pending.

For a fresh artifact, `prepare.py --root <checkout> --artifact <new-artifact>
--original <original-natural-ASR-artifact>` verifies every original frozen file,
copies the immutable reference sources, and proves that every native `main()`
syntax-tree node after platform setup remains identical. Run the saved supervisor
with `run --artifact <artifact> --engine native --family whisper --mode inputs`,
then `freeze.py --artifact <artifact> --source-commit <commit>` after those process
births are terminal. Supply the isolated package and existing psutil directories
through `PYTHONPATH`. Launch the frozen `runtime/supervise.py` with `launch
--artifact <artifact> --engine native` once. All these writers have already
completed for the current artifact and must not be restarted.

Full public decisions and the previous Windows first result will be compared,
with the same human-reference policy and no changed numerical tolerance. The
first dependency-check failure from Python-version-dependent `ast.dump` formatting
is retained; a separate portable tree comparison and exact generator/frontend
file hashes resolve that check without changing decoder code.
