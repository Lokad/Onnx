param(
    [Parameter(Mandatory)][string]$Output,
    [string]$Models = 'models/parakeet-tdt-0.6b-v3'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $destination = [IO.Path]::GetFullPath($Output)
    $modelDirectory = [IO.Path]::GetFullPath($Models)
    if (Test-Path -LiteralPath $destination) { throw "Output exists; choose a new directory: $destination" }
    foreach ($file in @('encoder-model.onnx', 'encoder-model.onnx.data', 'decoder_joint-model.onnx')) {
        if (-not (Test-Path -LiteralPath (Join-Path $modelDirectory $file))) { throw "Stage the pinned local Parakeet files first; missing $file" }
    }
    New-Item -ItemType Directory -Path $destination | Out-Null
    & dotnet build tests/parakeet/ParakeetReplay.csproj -c Release --tl:off --nologo -v minimal -p:UseSharedCompilation=false *> (Join-Path $destination 'build.log')
    if ($LASTEXITCODE -ne 0) { throw "Build failed; inspect $destination/build.log" }
    $fixtures = Join-Path $destination 'reference'
    & python tests/parakeet/generate_reference.py --models $modelDirectory --output $fixtures *> (Join-Path $destination 'reference.log')
    if ($LASTEXITCODE -ne 0) { throw "Native oracle failed; inspect $destination/reference.log" }
    & dotnet tests/parakeet/bin/Release/net10.0/ParakeetReplay.dll $modelDirectory (Join-Path $fixtures 'manifest.json') (Join-Path $destination 'managed.json') *> (Join-Path $destination 'managed.log')
    if ($LASTEXITCODE -ne 0) { throw "Managed replay failed; inspect $destination/managed.json and managed.log" }
    Write-Host "PASS Parakeet full-tensor component replay. Audio preprocessing and TDT transcription remain separate. Results: $destination"
} finally { Pop-Location }
