param(
    [string]$Python = "python",
    [ValidateSet('net10.0', 'net11.0')][string]$Framework = 'net11.0'
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Fail($msg) { Write-Host "FAIL opfuzz: $msg"; exit 1 }
function Invoke-GateNative {
    param([scriptblock]$Command)
    $prev = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Command
    } finally { $ErrorActionPreference = $prev }
}
$corpus = "tests/opfuzz/corpus"
$sums = Join-Path $corpus "SHA256SUMS"
if (!(Test-Path $sums)) { Fail "missing hash manifest: $sums" }
$expected = @{}
foreach ($line in (Get-Content $sums)) {
    if ($line.Trim() -eq "") { continue }
    $parts = $line.Split(" ", 2, [System.StringSplitOptions]::RemoveEmptyEntries)
    $expected[$parts[1]] = $parts[0]
}
$actual = Get-ChildItem $corpus -Recurse -File | Where-Object { $_.Name -ne "SHA256SUMS" }
if ($actual.Count -ne $expected.Count) { Fail "corpus file count changed (actual $($actual.Count), manifest $($expected.Count))" }
foreach ($f in $actual) {
    $rel = $f.FullName.Substring((Join-Path $root $corpus).Length + 1).Replace("\", "/")
    $h = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
    if (!$expected.ContainsKey($rel) -or $expected[$rel] -ne $h) { Fail "hash mismatch or unknown file: $rel" }
}
Write-Host "PASS assets: corpus hashes match ($($expected.Count) files)"
Invoke-GateNative { & dotnet build tests/Lokad.Onnx.OpDump -f $Framework --tl:off --nologo -v minimal -c Release -p:NuGetAudit=false }
if ($LASTEXITCODE -ne 0) { Fail "OpDump build failed" }
$savedRunner = $env:OPDUMP_DLL
try {
    $env:OPDUMP_DLL = Join-Path $root "tests/Lokad.Onnx.OpDump/bin/Release/$Framework/Lokad.Onnx.OpDump.dll"
    Invoke-GateNative { & $Python -m pytest tests/opfuzz/python -q --no-header }
    if ($LASTEXITCODE -ne 0) { Fail "opfuzz conformance suite failed" }
} finally { $env:OPDUMP_DLL = $savedRunner }
Write-Host "PASS opfuzz conformance"
