param([Parameter(Mandatory = $true)][string]$ManifestPath)

# Remove only the reviewed obsolete artifact files. Keep a hash receipt before
# each removal, and refuse changed files, tracked files, or reparse-point paths.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$artifactRoot = [IO.Path]::Combine($repoRoot, 'artifacts')
$artifactPrefix = $artifactRoot + [IO.Path]::DirectorySeparatorChar
$manifestFile = Get-Item -LiteralPath $ManifestPath
$manifest = Get-Content -LiteralPath $manifestFile.FullName -Raw | ConvertFrom-Json -AsHashtable
if ([IO.Path]::GetFullPath($manifest.repo) -ne $repoRoot) { throw 'Wrong repository' }
$manifestHash = (Get-FileHash -LiteralPath $manifestFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$receiptDirectory = $manifestFile.DirectoryName
if (-not $receiptDirectory.StartsWith($artifactPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Receipt outside artifacts' }
$journalPath = Join-Path $receiptDirectory 'retired.jsonl'
$completedPath = Join-Path $receiptDirectory 'completed.json'
if (Test-Path -LiteralPath $completedPath) { throw 'Already completed' }
$keep = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($name in $manifest.keep) { [void]$keep.Add($name) }
$tracked = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($name in (& git -C $repoRoot ls-files -- artifacts)) { [void]$tracked.Add($name) }
if ($LASTEXITCODE -ne 0) { throw 'Cannot check tracked files' }
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($entry in $manifest.files) {
    $target = [IO.Path]::GetFullPath([IO.Path]::Combine($repoRoot, $entry.path))
    $parts = $entry.path.Split('/')
    if (-not $target.StartsWith($artifactPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        $parts.Count -lt 3 -or $parts[0] -ne 'artifacts' -or $parts[1].Contains('20260923') -or
        $keep.Contains($parts[1]) -or $tracked.Contains($entry.path) -or -not $seen.Add($target)) {
        throw "Invalid retirement target: $($entry.path)"
    }
}
$prior = @{}
if (Test-Path -LiteralPath $journalPath) {
    foreach ($line in [IO.File]::ReadLines($journalPath)) {
        $entry = $line | ConvertFrom-Json -AsHashtable
        if ($entry.manifest_sha256 -ne $manifestHash) { throw 'Different retirement manifest' }
        $prior[$entry.path] = $entry
    }
}
$checkedDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$epochTicks = [DateTime]::UnixEpoch.Ticks
$writer = [IO.StreamWriter]::new($journalPath, $true, [Text.UTF8Encoding]::new($false))
$started = [DateTime]::UtcNow
$processed = 0L
$retiredBytes = 0L
$lastProgress = [DateTime]::UtcNow
$success = $false
try {
    foreach ($entry in $manifest.files) {
        $target = [IO.Path]::GetFullPath([IO.Path]::Combine($repoRoot, $entry.path))
        if ($prior.ContainsKey($entry.path) -and -not (Test-Path -LiteralPath $target)) {
            if ($prior[$entry.path].bytes -ne $entry.bytes) { throw 'Receipt size mismatch' }
            $processed++; $retiredBytes += $entry.bytes
            continue
        }
        $item = Get-Item -LiteralPath $target -Force
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Not a regular file: $target" }
        $directory = $item.Directory
        while ($directory -and $directory.FullName.StartsWith($repoRoot, [StringComparison]::OrdinalIgnoreCase)) {
            if ($checkedDirectories.Contains($directory.FullName)) { break }
            if ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse-point ancestor: $($directory.FullName)" }
            [void]$checkedDirectories.Add($directory.FullName)
            $directory = $directory.Parent
        }
        $mtimeNs = [long](($item.LastWriteTimeUtc.Ticks - $epochTicks) * 100L)
        if ($item.Length -ne $entry.bytes -or $mtimeNs -ne $entry.mtime_ns) { throw "File changed since inventory: $target" }
        $stream = [IO.File]::OpenRead($target)
        try { $digest = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($stream)) }
        finally { $stream.Dispose() }
        if ($prior.ContainsKey($entry.path)) {
            if ($prior[$entry.path].sha256 -ne $digest) { throw "Pending file changed: $target" }
        } else {
            $record = @{ path = $entry.path; bytes = $entry.bytes; mtime_ns = $entry.mtime_ns; sha256 = $digest; manifest_sha256 = $manifestHash }
            $writer.WriteLine(($record | ConvertTo-Json -Compress))
            $writer.Flush()
        }
        Remove-Item -LiteralPath $target -Force
        if (Test-Path -LiteralPath $target) { throw "Removal failed: $target" }
        $processed++; $retiredBytes += $entry.bytes
        if (([DateTime]::UtcNow - $lastProgress).TotalSeconds -ge 15) {
            Write-Output ("Retired {0:N0}/{1:N0} files, {2:N2} GB" -f $processed, $manifest.files.Count, ($retiredBytes / 1e9))
            $lastProgress = [DateTime]::UtcNow
        }
    }
    if ($processed -ne $manifest.files.Count -or $retiredBytes -ne $manifest.planned_bytes) { throw 'Incomplete retirement' }
    $success = $true
} finally {
    $writer.Dispose()
    $receipt = @{ passed = $success; manifest_sha256 = $manifestHash; processed = $processed; bytes = $retiredBytes; started_utc = $started.ToString('o'); ended_utc = [DateTime]::UtcNow.ToString('o'); pid = $PID }
    $receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $receiptDirectory 'latest.json') -Encoding utf8
    if ($success) { $receipt | ConvertTo-Json | Set-Content -LiteralPath $completedPath -Encoding utf8 }
}
Write-Output ("Completed: {0:N0} files, {1:N3} GB retired" -f $processed, ($retiredBytes / 1e9))
