param(
    [Parameter(Mandatory)][string]$Output,
    [string]$L0Ref = 'b2ffb909',
    [string]$L1Ref = 'HEAD'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$destination = [IO.Path]::GetFullPath($Output)
& ./eng/build-common-campaign.ps1 -Output $destination -L0Ref $L0Ref -L1Ref $L1Ref -CoreOnly
if ($LASTEXITCODE -ne 0) { throw 'Core preparation failed' }
$baseline = Join-Path $destination 'L0/core/Lokad.Onnx.dll'
dotnet build tests/Lokad.Onnx.EmbeddingGemma2Bench -c Release --tl:off --nologo -v minimal "-p:CoreAssemblyPath=$baseline" -o (Join-Path $destination 'runner') *> (Join-Path $destination 'runner-build.log')
if ($LASTEXITCODE -ne 0) { throw "Benchmark build failed: $destination/runner-build.log" }
Write-Host "PASS one shared EmbeddingGemma 2 runner and pinned cores: $destination"
