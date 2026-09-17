# Build pinned cores from local git archives and one shared runner. No VM access,
# fetch, worktree deletion or modification of another experiment's outputs.
param([Parameter(Mandatory)][string]$Output,
      [string]$L0Ref = "4495fc68b9505b0b6fab73146bd905424588218e",
      [string]$L1Ref = "HEAD")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$destination = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $destination) { throw "Output already exists; choose a new directory: $destination" }
New-Item -ItemType Directory -Path $destination | Out-Null
Push-Location $root
try {
    $sdk = (& dotnet --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Cannot resolve build SDK" }
    foreach ($role in @("L0", "L1")) {
        $reference = if ($role -eq "L0") { $L0Ref } else { $L1Ref }
        $revision = (& git rev-parse --verify --end-of-options "$reference^{commit}" | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $revision -notmatch '^[0-9a-f]{40}$') { throw "Invalid source revision $reference" }
        $leg = Join-Path $destination $role
        New-Item -ItemType Directory -Path $leg | Out-Null
        $archive = Join-Path $leg "source.zip"
        & git archive --format=zip --output=$archive $revision src/Lokad.Onnx
        if ($LASTEXITCODE -ne 0) { throw "git archive failed for $role" }
        $source = Join-Path $leg "source"
        Expand-Archive -LiteralPath $archive -DestinationPath $source
        Copy-Item -LiteralPath (Join-Path $root "global.json") -Destination (Join-Path $source "global.json")
        $binary = Join-Path $leg "core"
        & dotnet build (Join-Path $source "src/Lokad.Onnx/Lokad.Onnx.csproj") -c Release --tl:off --nologo -v minimal -p:UseSharedCompilation=false -p:EnableSourceControlManagerQueries=false "-p:SourceRevisionId=$revision" -o $binary *> (Join-Path $leg "build.log")
        if ($LASTEXITCODE -ne 0) { throw "Core build failed; inspect $leg/build.log" }
        $core = Join-Path $binary "Lokad.Onnx.dll"
        [ordered]@{
            source_sha = $revision; core_sha256 = (Get-FileHash -LiteralPath $core -Algorithm SHA256).Hash.ToLowerInvariant()
            core = "core/Lokad.Onnx.dll"; sdk = $sdk
            source_archive_sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $leg "build.json") -Encoding utf8
        Write-Host "$role built $revision with SDK $sdk"
    }
    $baseline = Join-Path $destination "L0/core/Lokad.Onnx.dll"
    $runner = Join-Path $destination "runner"
    & dotnet build tests/Lokad.Onnx.Campaign/Lokad.Onnx.Campaign.csproj -c Release --tl:off --nologo -v minimal "-p:CoreAssemblyPath=$baseline" -o $runner *> (Join-Path $destination "runner-build.log")
    if ($LASTEXITCODE -ne 0) { throw "Common runner build failed; inspect $destination/runner-build.log" }
    Write-Host "Prepared pinned cores and one common runner in $destination"
} finally { Pop-Location }
