# Build pinned cores from local git archives and one shared runner. No VM access,
# fetch, worktree deletion or modification of another experiment's outputs.
param([Parameter(Mandatory)][string]$Output,
      [Parameter(Mandatory)][string]$L0Ref,
      [string]$L1Ref = "HEAD",
      [switch]$CoreOnly,
      [ValidateSet('net10.0', 'net11.0')][string]$Framework = 'net11.0')
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$destination = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $destination) { throw "Output already exists; choose a new directory: $destination" }
Push-Location $root
try {
    $sources = @(
        foreach ($role in @('L0', 'L1')) {
            $reference = if ($role -eq 'L0') { $L0Ref } else { $L1Ref }
            $revision = (& git rev-parse --verify --end-of-options "$reference^{commit}" | Out-String).Trim()
            if ($LASTEXITCODE -ne 0 -or $revision -notmatch '^[0-9a-f]{40}$') { throw "Invalid source revision $reference" }
            $adapterPaths = @(& git ls-tree -d --name-only $revision -- tools/Lokad.Onnx.Protobuf)
            if ($LASTEXITCODE -ne 0) { throw "Cannot inspect adapter layout for $role" }
            $projectText = @(& git show "${revision}:src/Lokad.Onnx/Lokad.Onnx.csproj")
            if ($LASTEXITCODE -ne 0) { throw "Cannot inspect core framework for $role" }
            $project = [xml]($projectText -join "`n")
            $frameworks = @($project.SelectNodes('//TargetFramework | //TargetFrameworks') | ForEach-Object { $_.InnerText.Split(';') })
            if ($Framework -notin $frameworks) {
                throw "$role source $revision does not target $Framework. Choose a compatible source revision or explicitly select -Framework net10.0 for historical comparisons."
            }
            [pscustomobject]@{ Role=$role; Revision=$revision; HasAdapter=($adapterPaths.Count -gt 0) }
        }
    )
    if ($sources[0].HasAdapter -ne $sources[1].HasAdapter) { throw 'Cannot compare integrated and separate importer assemblies with one unchanged runner. Choose both refs from the same importer era.' }
    New-Item -ItemType Directory -Path $destination | Out-Null
    $sdk = (& dotnet --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Cannot resolve build SDK" }
    foreach ($sourceInfo in $sources) {
        $role = $sourceInfo.Role
        $revision = $sourceInfo.Revision
        $hasAdapter = $sourceInfo.HasAdapter
        $leg = Join-Path $destination $role
        New-Item -ItemType Directory -Path $leg | Out-Null
        $archive = Join-Path $leg "source.zip"
        $archivePaths = @('src/Lokad.Onnx')
        if ($hasAdapter) { $archivePaths += 'tools/Lokad.Onnx.Protobuf' }
        & git archive --format=zip --output=$archive $revision @archivePaths
        if ($LASTEXITCODE -ne 0) { throw "git archive failed for $role" }
        $source = Join-Path $leg "source"
        Expand-Archive -LiteralPath $archive -DestinationPath $source
        Copy-Item -LiteralPath (Join-Path $root "global.json") -Destination (Join-Path $source "global.json")
        $binary = Join-Path $leg "core"
        & dotnet build (Join-Path $source "src/Lokad.Onnx/Lokad.Onnx.csproj") -c Release -f $Framework --tl:off --nologo -v minimal -p:UseSharedCompilation=false -p:EnableSourceControlManagerQueries=false "-p:SourceRevisionId=$revision" -o $binary *> (Join-Path $leg "build.log")
        if ($LASTEXITCODE -ne 0) { throw "Core build failed; inspect $leg/build.log" }
        if ($hasAdapter) {
            & dotnet build (Join-Path $source "tools/Lokad.Onnx.Protobuf/Lokad.Onnx.Protobuf.csproj") -c Release -f $Framework --tl:off --nologo -v minimal -p:UseSharedCompilation=false -p:EnableSourceControlManagerQueries=false -p:CopyLocalLockFileAssemblies=true "-p:SourceRevisionId=$revision" -o $binary *> (Join-Path $leg "adapter-build.log")
            if ($LASTEXITCODE -ne 0) { throw "Adapter build failed; inspect $leg/adapter-build.log" }
        }
        $core = Join-Path $binary "Lokad.Onnx.dll"
        [ordered]@{
            source_sha = $revision; core_sha256 = (Get-FileHash -LiteralPath $core -Algorithm SHA256).Hash.ToLowerInvariant()
            core = "core/Lokad.Onnx.dll"; sdk = $sdk; target_framework = $Framework
            importer_layout = if ($hasAdapter) { 'separate' } else { 'integrated' }
            source_archive_sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $leg "build.json") -Encoding utf8
        Write-Host "$role built $revision with SDK $sdk"
    }
    if ($CoreOnly) {
        Write-Host "Prepared pinned core archives/binaries in $destination"
        return
    }
    $baseline = Join-Path $destination "L0/core/Lokad.Onnx.dll"
    $runner = Join-Path $destination "runner"
    $adapter = Join-Path $destination 'L0/core/Lokad.Onnx.Protobuf.dll'
    $adapterArguments = @()
    if (Test-Path -LiteralPath $adapter) { $adapterArguments += "-p:ProtobufAssemblyPath=$adapter" }
    & dotnet build tests/Lokad.Onnx.Campaign/Lokad.Onnx.Campaign.csproj -c Release -f $Framework --tl:off --nologo -v minimal "-p:CoreAssemblyPath=$baseline" @adapterArguments -o $runner *> (Join-Path $destination "runner-build.log")
    if ($LASTEXITCODE -ne 0) { throw "Common runner build failed; inspect $destination/runner-build.log" }
    Write-Host "Prepared pinned cores and one common runner in $destination"
} finally { Pop-Location }
