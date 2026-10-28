# Freeze each revision's complete managed application, then compile one API caller.
param(
    [Parameter(Mandatory)][string]$Output,
    [Parameter(Mandatory)][string]$L0Ref,
    [string]$L1Ref = 'HEAD',
    [ValidateSet('net10.0', 'net11.0')][string]$Framework = 'net11.0'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $destination = [IO.Path]::GetFullPath($Output)
    & ./eng/build-common-campaign.ps1 -Output $destination -L0Ref $L0Ref -L1Ref $L1Ref -Framework $Framework -CoreOnly
    if ($LASTEXITCODE -ne 0) { throw 'Core preparation failed' }
    foreach ($role in @('L0', 'L1')) {
        $leg = Join-Path $destination $role
        $receipt = Join-Path $leg 'build.json'
        $build = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json -AsHashtable
        if ($build.importer_layout -ne 'separate') { throw 'Audio comparisons require the separate Protobuf adapter.' }
        $archive = Join-Path $leg 'data-source.zip'
        & git archive --format=zip --output=$archive $build.source_sha src/Lokad.Onnx.Data
        if ($LASTEXITCODE -ne 0) { throw "Cannot archive $role Data sources" }
        Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $leg 'source') -Force
        $binary = Join-Path $leg 'core'
        & dotnet build (Join-Path $leg 'source/src/Lokad.Onnx.Data/Lokad.Onnx.Data.csproj') -c Release -f $Framework --tl:off --nologo -v minimal -p:UseSharedCompilation=false -p:EnableSourceControlManagerQueries=false -p:CopyLocalLockFileAssemblies=true "-p:SourceRevisionId=$($build.source_sha)" -o $binary *> (Join-Path $leg 'data-build.log')
        if ($LASTEXITCODE -ne 0) { throw "Data build failed; inspect $leg/data-build.log" }
        $coreHash = (Get-FileHash (Join-Path $binary 'Lokad.Onnx.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($coreHash -ne $build.core_sha256) { throw "Data build changed the qualified $role core" }
        $build.data_source_archive_sha256 = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        $build.product_files = @{}
        foreach ($file in Get-ChildItem -LiteralPath $binary -Filter '*.dll') {
            $build.product_files[$file.Name] = (Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        $build | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receipt -Encoding utf8
    }
    $baseline = Join-Path $destination 'L0/core'
    & dotnet build tests/Lokad.Onnx.AudioBench -c Release -f $Framework --tl:off --nologo -v minimal "-p:ProductDirectory=$baseline" -o (Join-Path $destination 'runner') *> (Join-Path $destination 'runner-build.log')
    if ($LASTEXITCODE -ne 0) { throw "Audio caller build failed; inspect $destination/runner-build.log" }
    Write-Host "PASS one audio caller and pinned complete products: $destination"
} finally { Pop-Location }
