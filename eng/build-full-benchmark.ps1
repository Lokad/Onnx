# Build the three small callers against one already qualified current product.
param(
    [Parameter(Mandatory)][string]$Product,
    [Parameter(Mandatory)][string]$Output,
    [string]$SourceRef = 'HEAD'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $productPath = [IO.Path]::GetFullPath($Product)
    $destination = [IO.Path]::GetFullPath($Output)
    if (Test-Path -LiteralPath $destination) { throw 'Choose a fresh full benchmark output directory' }
    $build = Get-Content -LiteralPath (Join-Path $productPath 'build.json') -Raw | ConvertFrom-Json -AsHashtable
    $framework = $build.target_framework
    if ($framework -notin @('net10.0', 'net11.0')) { throw 'Full benchmark needs a qualified .NET 10 or .NET 11 product' }
    $revision = (& git rev-parse --verify --end-of-options "$SourceRef^{commit}" | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $revision -notmatch '^[0-9a-f]{40}$') { throw 'Invalid full caller source revision' }
    $paths = @('global.json', 'eng', 'tests/Lokad.Onnx.AudioBench', 'tests/Lokad.Onnx.GraphBench',
        'tests/Lokad.Onnx.EmbeddingGemma2Bench', 'tests/Shared/NpySupport.cs',
        'tests/Lokad.Onnx.Bench/CampaignEvidence.cs', 'tests/Lokad.Onnx.Bench/SampleDiagnostics.cs', 'src/Lokad.Onnx.Data')
    foreach ($path in $paths) {
        & git cat-file -e "${revision}:$path"
        if ($LASTEXITCODE -ne 0) { throw "Commit the full caller sources before preparation: $path" }
    }
    & git diff --quiet $build.source_sha $revision -- 'src/Lokad.Onnx.Data/EmbeddingGemma2*.cs'
    if ($LASTEXITCODE -ne 0) { throw 'Gemma caller/Data sources differ from the qualified product' }
    $sdk = (& dotnet --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $sdk -ne $build.sdk) { throw 'Use the same SDK as the qualified product' }
    $binary = Join-Path $productPath 'core'
    foreach ($name in @('source.zip', 'data-source.zip')) {
        $field = if ($name -eq 'source.zip') { 'source_archive_sha256' } else { 'data_source_archive_sha256' }
        if ((Get-FileHash (Join-Path $productPath $name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $build[$field]) {
            throw "Qualified product source changed: $name"
        }
    }
    foreach ($name in $build.product_files.Keys) {
        if ([IO.Path]::GetFileName($name) -ne $name -or -not $name.EndsWith('.dll')) { throw 'Invalid product filename' }
        if ((Get-FileHash (Join-Path $binary $name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $build.product_files[$name]) {
            throw "Qualified product changed: $name"
        }
    }
    New-Item -ItemType Directory -Path $destination | Out-Null
    $leg = Join-Path $destination 'L1'
    New-Item -ItemType Directory -Path $leg | Out-Null
    foreach ($name in @('build.json', 'source.zip', 'data-source.zip')) {
        Copy-Item -LiteralPath (Join-Path $productPath $name) -Destination $leg
    }
    Copy-Item -LiteralPath $binary -Destination (Join-Path $leg 'core') -Recurse
    $archive = Join-Path $destination 'runner-source.zip'
    & git archive --format=zip --output=$archive $revision @paths
    if ($LASTEXITCODE -ne 0) { throw 'Cannot archive full benchmark caller sources' }
    $source = Join-Path $destination 'source'
    Expand-Archive -LiteralPath $archive -DestinationPath $source
    $runner = Join-Path $destination 'runner'
    $productArgument = "-p:ProductDirectory=$binary"
    foreach ($project in @('Lokad.Onnx.AudioBench', 'Lokad.Onnx.GraphBench', 'Lokad.Onnx.EmbeddingGemma2Bench')) {
        [string[]]$arguments = if ($project -eq 'Lokad.Onnx.EmbeddingGemma2Bench') {
            @("-p:CoreAssemblyPath=$(Join-Path $binary 'Lokad.Onnx.dll')", "-p:GemmaDataSourcePath=$(Join-Path $source 'src/Lokad.Onnx.Data')")
        } else { @($productArgument) }
        & dotnet build (Join-Path $source "tests/$project/$project.csproj") -c Release -f $framework --tl:off --nologo -v minimal -p:UseSharedCompilation=false "-p:SourceRevisionId=$revision" @arguments -o $runner *> (Join-Path $destination "$project-build.log")
        if ($LASTEXITCODE -ne 0) { throw "Full caller build failed: $project" }
    }
    foreach ($name in $build.product_files.Keys) {
        Copy-Item -LiteralPath (Join-Path $binary $name) -Destination (Join-Path $runner $name) -Force
    }
    $files = @{}
    foreach ($file in Get-ChildItem -LiteralPath $runner -File -Recurse) {
        $name = [IO.Path]::GetRelativePath($runner, $file.FullName).Replace([IO.Path]::DirectorySeparatorChar, [char]'/')
        $files[$name] = (Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    [ordered]@{ source_sha=$revision; sdk=$sdk; target_framework=$framework; files=$files
        source_archive_sha256=(Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    } | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $destination 'full-build.json') -Encoding utf8NoBOM
    Write-Host "PASS full benchmark callers against unchanged qualified product: $destination"
} finally { Pop-Location }
