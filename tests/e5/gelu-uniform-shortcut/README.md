# Exact conditional GELU prototype

The [complete e5 census](../gelu-branch-census/results-20260920.md) found
10.31–12.01% all-small eight-lane vectors on the primary inputs. This standalone
prototype skips the unused large-value polynomial/exponential when every lane
selects the existing small-value polynomial. Mixed vectors keep the original
arithmetic. No layer name, model identity or approximate formula selects the path.

The first local proof **passes 1,575 cases and 102,364,884 value comparisons**.
That total counts three variants (two unchanged copies and the conditional
candidate), each out-of-place and in-place; it is not a count of unique inputs.
The candidate accounts for 34,121,628 comparisons. Every result matches the
actual qualified product kernel's bytes. Tests include all sixty retained
model captures, 2,097,152 random float bit patterns, threshold neighbors,
mixed NaN payloads/infinities/subnormals/signed zero, geometry/fallback cases,
offset spans, input/bias preservation and destination sentinels.

[Proof records and identities](proof-20260920.json) retain the result. Source is
`cf325d7`; product core remains
`187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4`.
This is Windows .NET10.0.12, CPU2, eight-lane vectors and FMA, without AVX-512.
Build uses SDK10.0.204 and passes with zero warnings/errors. Sampled process peak
is 167,706,624 bytes across five observations; both recorded process births are
absent. Independent verification checks all frozen files, the full upstream
census receipt, sixty input identities and the exact case/comparison accounting.
Raw proof and closure are in `artifacts/gelu-uniform-shortcut-20260920`.

**No timing, AMD qualification or production change is claimed.** The next
experiment must price predicate/branch overhead on every mixed vector, use the
complete twelve-layer input banks and duplicate controls, and retain actual AMD
code and resource evidence. A kernel win would still need product and model
qualification.

Generate into a new directory, then build against the qualified core:

```powershell
python -X utf8 -B tests/e5/gelu-uniform-shortcut/generate.py --output <new-generated-directory>
dotnet build <new-generated-directory>/Probe.csproj -c Release -p:FrozenCorePath=<qualified-core.dll> -o <new-bin-directory> --tl:off --nologo -v minimal
dotnet <new-bin-directory>/Probe.dll artifacts/gelu-branch-census-20260920 <new-proof.json>
```

The proof requires actual eight-lane FMA hardware and verifies every file in the
closed upstream census before reading captures. Output paths are single-use;
preserve failures. This executable performs arithmetic proof only. It includes
no benchmark loop or product dispatch.
