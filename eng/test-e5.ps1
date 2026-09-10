param(
    [string]$Python = "python",
    [switch]$RequireIntrinsics
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
# Determinism: disable tiered-JIT On-Stack Replacement so repeated runs are
# bit-stable. OSR can swap loop code mid-flight (e.g. FMA contraction), which
# moves last-ulp results run to run under contention; verified: scalar repeats
# drift ~1e-05 in a few dozen cells with OSR on and are byte-identical with it
# off. Tolerances remain the correctness bar; this only stabilizes repeats.
# The perf lane (bench.ps1) intentionally keeps default JIT settings.
$env:DOTNET_JitOSR = '0'
$model = "models/multilingual-e5-small/model.onnx"
$tokenizer = "models/multilingual-e5-small/sentencepiece.bpe.model"
$cases = "tests/e5/cases.json"
$artifacts = "artifacts/e5"
$expectedModel = "CA456C06B3A9505DDFD9131408916DD79290368331E7D76BB621F1CBA6BC8665"
$expectedTokenizer = "CFC8146ABE2A0488E9E2A0C56DE7952F7C11AB059ECA145A0A727AFCE0DB2865"
function Fail($msg) { Write-Host "FAIL e5: $msg"; exit 1 }
function Invoke-GateNative {
    param([scriptblock]$Command)
    $prev = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Command
    } finally { $ErrorActionPreference = $prev }
}
if (!(Test-Path $model)) { Fail "missing model asset: $model" }
if (!(Test-Path $tokenizer)) { Fail "missing tokenizer asset: $tokenizer" }
if ((Get-FileHash $model -Algorithm SHA256).Hash -ne $expectedModel) { Fail "model hash mismatch" }
if ((Get-FileHash $tokenizer -Algorithm SHA256).Hash -ne $expectedTokenizer) { Fail "tokenizer hash mismatch" }
Write-Host "PASS assets: model and tokenizer hashes match"
$pyv = & $Python --version 2>&1
Write-Host "Python: $pyv"
$prevEAP = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $Python -c "import numpy, onnxruntime, transformers, sentencepiece, pytest" 2>&1 | Out-Null
    $probeEc = $LASTEXITCODE
} finally { $ErrorActionPreference = $prevEAP }
if ($probeEc -ne 0) { Fail "python is missing test dependencies (see tests/e5/python/requirements.txt)" }
Invoke-GateNative { & dotnet build tests/Lokad.Onnx.E5Runner --tl:off --nologo -v minimal -c Release -p:NuGetAudit=false }
if ($LASTEXITCODE -ne 0) { Fail "E5Runner build failed" }
$runner = "tests/Lokad.Onnx.E5Runner/bin/Release/net10.0/Lokad.Onnx.E5Runner.dll"
New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
$modes = @("scalar", "simd", "intrinsics")
foreach ($m in $modes) {
    $first = "$artifacts/dotnet-$m.json"
    $repeat = "$artifacts/dotnet-$m.repeat.json"
    Invoke-GateNative { & dotnet $runner --model $model --tokenizer $tokenizer --cases $cases --mode $m --output $first }
    if ($LASTEXITCODE -ne 0) {
        if ($m -eq "intrinsics" -and !$RequireIntrinsics) {
            Write-Host "UNSUPPORTED intrinsics: x86 FMA not available on this machine (rerun with -RequireIntrinsics to enforce)"
            Remove-Item $first -Force -ErrorAction SilentlyContinue
            Remove-Item $repeat -Force -ErrorAction SilentlyContinue
            '{"schemaVersion": 1, "unsupported": "x86 FMA not available on this machine"}' | Set-Content -NoNewline -Encoding utf8NoBOM $first
            continue
        }
        Fail "E5Runner failed for mode $m"
    }
    Invoke-GateNative { & dotnet $runner --model $model --tokenizer $tokenizer --cases $cases --mode $m --output $repeat }
    if ($LASTEXITCODE -ne 0) { Fail "E5Runner repeat failed for mode $m" }
    $h1 = (Get-FileHash $first -Algorithm SHA256).Hash
    $h2 = (Get-FileHash $repeat -Algorithm SHA256).Hash
    if ($h1 -ne $h2) { Fail "nondeterministic artifacts for mode $m" }
    Remove-Item $repeat -Force
    Write-Host "PASS ${m}: deterministic artifact"
}
Invoke-GateNative { & $Python -m pytest tests/e5/python/test_e5_conformance.py -q --no-header }
if ($LASTEXITCODE -ne 0) { Fail "native conformance suite failed" }
Write-Host "PASS e5 conformance"
