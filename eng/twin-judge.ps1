param([string]$Twin = ".agent/twin.log")
$pat = '^(\d+):(\d+):(\d+)\.(\d+) (\d+) (\S+) u=(\d+) k=(\d+)'
$first = @{}; $last = @{}; $nm = @{}; $t0 = -1; $t1 = -1; $lines = 0
$prev = @{}; $buckets = @{}
foreach ($l in [IO.File]::ReadLines($Twin))
{
    if ($l -notmatch $pat) { continue }
    $sec = [int]$Matches[1] * 3600 + [int]$Matches[2] * 60 + [int]$Matches[3]
    $p = $Matches[5]; $n = $Matches[6]; $tot = [long]$Matches[7] + [long]$Matches[8]
    if ($t0 -lt 0) { $t0 = $sec }; $t1 = $sec; $lines++
    if (-not $first.ContainsKey($p)) { $first[$p] = $tot; $nm[$p] = $n }
    $last[$p] = $tot
    if ($n -eq "MsMpEng.exe")
    {
        if ($prev.ContainsKey($p))
        {
            $b = [string]([int]($sec / 30)); if (-not $buckets.ContainsKey($b)) { $buckets[$b] = 0.0 }
            $buckets[$b] += ($tot - $prev[$p]) / 1e7
        }
        $prev[$p] = $tot
    }
}
echo "twin=$Twin lines=$lines window=$t0..$t1 ($($t1 - $t0)s)"
$by = @{}
foreach ($p in $first.Keys) { $n = $nm[$p]; if (-not $by.ContainsKey($n)) { $by[$n] = 0.0 }; $by[$n] += ($last[$p] - $first[$p]) / 1e7 }
$by.GetEnumerator() | Sort-Object Value -Descending | Format-Table -AutoSize
echo "--- MsMpEng CPU-s per 30s bucket ---"
$buckets.GetEnumerator() | Sort-Object Name | Format-Table -AutoSize
echo "rule: judge windows by twin totals plus distributions, never by live spot readings; sustained Defender burn through timed iterations means DISCARD."
