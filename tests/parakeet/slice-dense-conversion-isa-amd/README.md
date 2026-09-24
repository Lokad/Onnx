# Complete the missing instruction-mode qualification

The corrected test consumer passes all395 cases in ordinary mode. The intended
disabled-mode run passes394 cases but its mode assertion finds AVX512 still
enabled. Its DOTNET_EnableAVX512F flag was ineffective on this runtime. Preserve
that failed run; use DOTNET_EnableAVX512=0, already used by the normal release
protocol, for the missing disabled-mode qualification.

Run only that suite in a fresh namespace. Reuse exact Core49c3a958 and test
consumera17057a3 with no rebuild, source change, new numerical allowance or
repeat of the successful ordinary-mode run. The existing identity test must
observe Avx512F.IsSupported false and Fma.IsSupported true. The joint audit
retains every original case and joins the successful395-case suites by name.

Use run.py prepare/stage/launch/observe/collect, then audit.py. All owners and
files remain single-use, tools freeze at prepare, and original CPU, time, memory
and storage limits remain. Use local Python3.13 -X utf8 -B and exclusive AMD
execution with .NET10.0.204/runtime10.0.8 and --tl:off.
