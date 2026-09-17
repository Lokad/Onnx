# Process accounting shared by eng/run-l0l1-canonical.ps1 and
# eng/test_proc_accounting.ps1. DOT-SOURCE ONLY - this file must not execute
# anything on import (the lane runs top to bottom when sourced).
$script:ProcParseErrors = 0
$script:ProcErrAtPre = 0
function Parse-ProcLines([string[]]$lines) {
  # Linux ps -eo pid,ppid,times,etime,comm rows. times is BARE INTEGER
  # cumulative CPU seconds (M01: the old hh:mm:ss assumption read every row
  # as zero). Unparseable rows are SKIPPED and counted, never zero-filled:
  # the lane aborts release evidence when the count moves mid-rep.
  $rows = @()
  foreach ($line in $lines) {
    $p = $line.Trim() -split "\s+", 5
    if ($p.Count -lt 5) { $script:ProcParseErrors++; continue }
    $cid = 0; $cpp = 0; $cpu = -1.0
    if (-not ([int]::TryParse($p[0], [ref]$cid))) { $script:ProcParseErrors++; continue }
    if (-not ([int]::TryParse($p[1], [ref]$cpp))) { $script:ProcParseErrors++; continue }
    if (-not ([double]::TryParse($p[2], [ref]$cpu))) { $script:ProcParseErrors++; continue }
    $rows += [pscustomobject]@{ Id = $cid; Name = $p[4]; PPid = $cpp; CPUSec = $cpu; Elapsed = $p[3] }
  }
  return $rows
}
function Get-ProcTable {
  # Rows with Id, Name, PPid, CPUSec (cumulative CPU seconds), Elapsed
  # (start identity, informational only) on either OS.
  if ($IsLinux) {
    return Parse-ProcLines (& ps -eo pid=,ppid=,times=,etime=,comm= --no-headers 2>$null)
  } else {
    $rows = @()
    foreach ($pr in (Get-CimInstance Win32_Process)) {
      $cpu = ([double]$pr.KernelModeTime + [double]$pr.UserModeTime) / 10000000.0
      $rows += [pscustomobject]@{ Id = $pr.ProcessId; Name = $pr.Name; PPid = $pr.ParentProcessId; CPUSec = $cpu; Elapsed = "" }
    }
    return $rows
  }
}
function Test-LaneOwned($id, $map) {
  $seen = @{}
  $cur = $id
  while ($cur -gt 0 -and !$seen.ContainsKey($cur)) {
    if ($cur -eq $PID) { return $true }
    $seen[$cur] = 1
    if (!$map.ContainsKey($cur)) { return $false }
    $cur = $map[$cur][1]
  }
  return $false
}
