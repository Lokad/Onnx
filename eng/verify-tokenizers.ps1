param(
    [Parameter(Mandatory)][string]$PackagePath,
    [Parameter(Mandatory)][string]$AssetsPath
)
$ErrorActionPreference = 'Stop'
$assets = Get-Content -LiteralPath $AssetsPath -Raw | ConvertFrom-Json -AsHashtable
$library = $assets.libraries['Lokad.Tokenizers/0.2.0']
if (!$library -or $library.type -ne 'package' -or $library.path -ne 'lokad.tokenizers/0.2.0') {
    throw 'The restored assets must resolve Lokad.Tokenizers 0.2.0.'
}
$resolved = $null
foreach ($folder in $assets.packageFolders.Keys) {
    $candidate = Join-Path $folder $library.path
    if (Test-Path -LiteralPath $candidate -PathType Container) { $resolved = $candidate; break }
}
if (!$resolved) { throw 'Cannot locate the resolved Lokad.Tokenizers package.' }

# NuGet generates new archive metadata identifiers on every pack. Compare the
# actual assembly and dependency manifest, whose bytes are stable for the build.
$zip = [IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($PackagePath))
try {
    $hashes = @{}
    foreach ($entryName in @('lib/net10.0/Lokad.Tokenizers.dll', 'Lokad.Tokenizers.nuspec')) {
        $entry = $zip.GetEntry($entryName)
        if (!$entry) { throw "The qualified package is missing $entryName." }
        $stream = $entry.Open()
        try { $expected = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant() }
        finally { $stream.Dispose() }
        $relative = if ($entryName.EndsWith('.nuspec')) { $entryName.ToLowerInvariant() } else { $entryName }
        $path = Join-Path $resolved $relative
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "The resolved package is missing $relative." }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "Resolved Lokad.Tokenizers content differs from the pinned build: $path. Retry with NUGET_PACKAGES set to a fresh directory; the existing cache has not been modified."
        }
        $hashes[$entryName] = $actual
    }
    [ordered]@{
        packageSha256 = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash.ToLowerInvariant()
        resolvedPackageDirectory = [IO.Path]::GetFullPath($resolved)
        assemblySha256 = $hashes['lib/net10.0/Lokad.Tokenizers.dll']
        manifestSha256 = $hashes['Lokad.Tokenizers.nuspec']
    } | ConvertTo-Json -Compress
}
finally { $zip.Dispose() }
