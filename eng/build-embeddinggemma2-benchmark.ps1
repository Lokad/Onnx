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
# Compile the shared API consumer against the same revision as the control core.
# Current Data sources may use additive APIs unavailable in that older core.
$control = (Get-Content -LiteralPath (Join-Path $destination 'L0/build.json') -Raw | ConvertFrom-Json).source_sha
$dataArchive = Join-Path $destination 'data-source.zip'
git archive --format=zip --output=$dataArchive $control src/Lokad.Onnx.Data
if ($LASTEXITCODE -ne 0) { throw 'Could not archive control Data sources' }
$sharedSource = Join-Path $destination 'shared-source'
Expand-Archive -LiteralPath $dataArchive -DestinationPath $sharedSource
$dataSource = Join-Path $sharedSource 'src/Lokad.Onnx.Data'
dotnet build tests/Lokad.Onnx.EmbeddingGemma2Bench -c Release --tl:off --nologo -v minimal "-p:CoreAssemblyPath=$baseline" "-p:GemmaDataSourcePath=$dataSource" -o (Join-Path $destination 'runner') *> (Join-Path $destination 'runner-build.log')
if ($LASTEXITCODE -ne 0) { throw "Benchmark build failed: $destination/runner-build.log" }
Write-Host "PASS one shared EmbeddingGemma 2 runner and pinned cores: $destination"
