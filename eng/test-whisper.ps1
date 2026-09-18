param(
    [Parameter(Mandatory)][string]$Output,
    [string]$Models = 'models/whisper-large-v3-turbo'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $destination = [IO.Path]::GetFullPath($Output)
    $modelDirectory = [IO.Path]::GetFullPath($Models)
    if (Test-Path -LiteralPath $destination) { throw "Output exists; choose a new directory: $destination" }
    foreach ($file in @('onnx/decoder_model.onnx', 'onnx/decoder_with_past_model.onnx', 'generation_config.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $modelDirectory $file))) { throw "Stage the pinned local Whisper files first; missing $file" }
    }
    New-Item -ItemType Directory -Path $destination | Out-Null
    & dotnet build tests/whisper/DecoderReplay.csproj -c Release --tl:off --nologo -v minimal -p:UseSharedCompilation=false *> (Join-Path $destination 'build.log')
    if ($LASTEXITCODE -ne 0) { throw "Build failed; inspect $destination/build.log" }
    $fixtures = Join-Path $destination 'reference'
    & python tests/whisper/generate_decoder_reference.py --models $modelDirectory --output $fixtures *> (Join-Path $destination 'reference.log')
    if ($LASTEXITCODE -ne 0) { throw "Native oracle failed; inspect $destination/reference.log" }
    & dotnet tests/whisper/bin/Release/net10.0/DecoderReplay.dll $modelDirectory (Join-Path $fixtures 'manifest.json') (Join-Path $destination 'managed.json') *> (Join-Path $destination 'managed.log')
    if ($LASTEXITCODE -ne 0) { throw "Managed replay failed; inspect $destination/managed.json and managed.log" }
    Write-Host "PASS Whisper decoder component replay. Encoder, preprocessing and transcription are separate qualifications. Results: $destination"
} finally { Pop-Location }
