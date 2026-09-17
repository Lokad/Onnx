"""Read-only process snapshots for the common-runner supervisor.

Uses stable birth identities, not elapsed age. Linux fields are documented at
https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html . A pair of snapshots
cannot observe all CPU used by processes that disappear; this limitation is
reported, never interpreted as proof that a machine was exclusively idle.
"""
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def checked_rows(rows):
    result = {}
    for row in rows:
        if row.get("id") == 0:  # Windows Idle; not foreign application work.
            continue
        for key in ("id", "parent", "start"):
            if type(row.get(key)) is not int or row[key] < (1 if key == "id" else 0):
                raise ValueError("invalid process " + key)
        cpu = row.get("cpu")
        if type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0:
            raise ValueError("invalid cumulative process CPU")
        if row["id"] in result:
            raise ValueError("duplicate process ID")
        result[row["id"]] = row
    if not result:
        raise ValueError("empty process snapshot")
    return result


def linux_row(line, ticks):
    # comm may contain spaces and parentheses; the fields start after its last ).
    left, right = line.find("("), line.rfind(")")
    if left <= 0 or right <= left or ticks <= 0:
        raise ValueError("malformed /proc stat")
    fields = line[right + 1:].split()  # state is field 3, index 0.
    if len(fields) < 20:
        raise ValueError("truncated /proc stat")
    user, system = int(fields[11]), int(fields[12])
    if user < 0 or system < 0:
        raise ValueError("negative process CPU ticks")
    return {"id": int(line[:left].strip()), "parent": int(fields[1]),
            "start": int(fields[19]), "cpu": (user + system) / ticks,
            "name": line[left + 1:right]}


def snapshot():
    vanished = 0
    if sys.platform.startswith("linux"):
        ticks = os.sysconf("SC_CLK_TCK")
        rows = []
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                rows.append(linux_row((directory / "stat").read_text(), ticks))
            except (FileNotFoundError, ProcessLookupError):
                vanished += 1  # Process vanished during enumeration; retain the limitation.
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    elif sys.platform == "win32":
        script = r"""
$ErrorActionPreference = 'Stop'
Get-CimInstance Win32_Process | ForEach-Object {
    [pscustomobject]@{
        id = $_.ProcessId; parent = $_.ParentProcessId; name = $_.Name
        start = $(if ($null -ne $_.CreationDate) { $_.CreationDate.ToUniversalTime().Ticks } else { $null })
        cpu = $(if ($null -ne $_.KernelModeTime -and $null -ne $_.UserModeTime) {
            ([double]$_.KernelModeTime + [double]$_.UserModeTime) / 10000000.0
        } else { $null })
    }
} | ConvertTo-Json -Compress
"""
        process = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
                                 capture_output=True, text=True, check=True, timeout=30,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        rows = json.loads(process.stdout)
        boot = "windows-absolute-creation-ticks"
    else:
        raise ValueError("process accounting supports Windows and Linux only")
    table = checked_rows(rows)
    if os.getpid() not in table:
        raise ValueError("supervisor missing from process snapshot")
    return {"monotonic": time.monotonic(), "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "boot": boot, "logical_cpus": os.cpu_count(), "vanished": vanished,
            "processes": list(table.values())}


def owned(pid, table, root):
    seen = set()
    while pid in table and pid not in seen:
        row = table[pid]
        if (pid, row["start"]) == root:
            return True
        seen.add(pid)
        parent = table.get(row["parent"])
        # A recycled parent PID born after its child cannot establish ownership.
        if parent is None or parent["start"] > row["start"]:
            return False
        pid = parent["id"]
    return False


def foreign_fraction(before, after, root_pid):
    old, new = checked_rows(before["processes"]), checked_rows(after["processes"])
    count = before["logical_cpus"]
    elapsed = after["monotonic"] - before["monotonic"]
    if before["boot"] != after["boot"] or type(count) is not int or count <= 0 or count != after["logical_cpus"]:
        raise ValueError("CPU/boot identity changed")
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid accounting interval")
    if root_pid not in old or root_pid not in new or old[root_pid]["start"] != new[root_pid]["start"]:
        raise ValueError("supervisor identity changed")
    root = (root_pid, old[root_pid]["start"])
    foreign = 0.0
    for pid, row in new.items():
        if owned(pid, new, root):
            continue
        previous = old.get(pid)
        if previous is not None and previous["start"] == row["start"]:
            delta = row["cpu"] - previous["cpu"]
            if delta < 0:
                raise ValueError("CPU counter went backwards without a new start identity")
        else:
            delta = row["cpu"]  # New/reused PID: conservatively count its full lifetime.
        foreign += delta
    if not math.isfinite(foreign):
        raise ValueError("invalid foreign CPU sum")
    disappeared = sum(not owned(pid, old, root) and (pid not in new or row["start"] != new[pid]["start"])
                      for pid, row in old.items())
    return {"valid": True, "foreign_cpu_fraction": foreign / (elapsed * count),
            "foreign_cpu_seconds": foreign, "wall_seconds": elapsed, "logical_cpus": count,
            "exited_foreign_processes": disappeared,
            "vanished_during_snapshots": before.get("vanished", 0) + after.get("vanished", 0),
            "limitation": "snapshot deltas miss some CPU of exited/short-lived processes; stability gates still required"}
