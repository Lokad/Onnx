param(
    [string]$Python = 'python',
    [string]$ModelDirectory = 'models/embeddinggemma-2',
    [string]$Artifacts = 'artifacts/embeddinggemma2',
    [string[]]$Modes = @('scalar', 'simd', 'default'),
    [switch]$SkipBuild,
    [switch]$FrozenInputsOnly,
    [ValidateSet('net10.0', 'net11.0')][string]$Framework = 'net11.0'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
function Invoke-Checked([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Conformance step failed with exit code $LASTEXITCODE" }
}
foreach ($mode in $Modes) {
    if ($mode -notin @('scalar', 'simd', 'default')) { throw "Unknown mode: $mode" }
}
if ($Modes.Count -eq 0) { throw 'At least one execution mode is required.' }
$saved = @{}
foreach ($name in @('DOTNET_TieredCompilation', 'DOTNET_TieredPGO', 'DOTNET_JitOSR', 'DOTNET_TC_OnStackReplacement')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name)
    [Environment]::SetEnvironmentVariable($name, '0')
}
try {
    $reference = Join-Path $Artifacts 'reference'
    $prepared = Join-Path $ModelDirectory 'prepared-fp32'
    Invoke-Checked { & $Python tests/embeddinggemma2/python/reference.py --model-dir $ModelDirectory --cases tests/embeddinggemma2/cases.json --output $reference }
    Invoke-Checked { & $Python eng/prepare-embeddinggemma2.py --source $ModelDirectory --output $prepared --verify --reference $reference }
    if (!$SkipBuild) {
        Invoke-Checked { & ./eng/restore-tokenizers.ps1 }
        Invoke-Checked { dotnet build tests/Lokad.Onnx.EmbeddingGemma2Runner -c Release -f $Framework --tl:off --nologo -v minimal }
        if (!$FrozenInputsOnly) { Invoke-Checked { dotnet build src/Lokad.Onnx.CLI -c Release -f $Framework --tl:off --nologo -v minimal } }
    }
    $runner = "tests/Lokad.Onnx.EmbeddingGemma2Runner/bin/Release/$Framework/Lokad.Onnx.EmbeddingGemma2Runner.dll"
    foreach ($mode in $Modes) {
        $first = Join-Path $Artifacts "managed-$mode"
        $repeat = Join-Path $Artifacts "managed-$mode-repeat"
        Invoke-Checked { dotnet $runner --model (Join-Path $prepared 'model.onnx') --reference $reference --output $first --mode $mode }
        Invoke-Checked { & $Python tests/embeddinggemma2/python/verify_managed.py --reference $reference --actual $first }
        Invoke-Checked { dotnet $runner --model (Join-Path $prepared 'model.onnx') --reference $reference --output $repeat --mode $mode }
        Invoke-Checked { & $Python tests/embeddinggemma2/python/verify_managed.py --reference $reference --actual $repeat }
        foreach ($path in Get-ChildItem -LiteralPath $first -Filter '*.npy' -Recurse -File) {
            $relative = [IO.Path]::GetRelativePath([IO.Path]::GetFullPath($first), $path.FullName)
            $other = Join-Path $repeat $relative
            if ((Get-FileHash -LiteralPath $path.FullName).Hash -ne (Get-FileHash -LiteralPath $other).Hash) {
                throw "Repeatability failed: $mode/$relative"
            }
        }
        Write-Host "PASS $mode conformance and byte repeatability"
    }
    if (!$FrozenInputsOnly) {
        $api = Join-Path $Artifacts 'api'
        Invoke-Checked { dotnet $runner api $prepared tests/embeddinggemma2/cases.json $reference $api }
        Invoke-Checked { & $Python tests/embeddinggemma2/python/verify_api.py --reference $reference --actual $api --cases tests/embeddinggemma2/cases.json }
        Invoke-Checked { & $Python tests/embeddinggemma2/python/verify_cli.py --cli "src/Lokad.Onnx.CLI/bin/Release/$Framework/Lokad.Onnx.CLI.dll" --model-dir $prepared --reference $reference --cases tests/embeddinggemma2/cases.json --output (Join-Path $Artifacts 'cli') }
    }
} finally {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
Write-Host 'PASS EmbeddingGemma 2 conformance'
