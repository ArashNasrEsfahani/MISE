#!/bin/bash
# K23 process ownership helpers (Astra K23-PREP r4 3-4, r5 3-4), SOURCED by capture_one_k23.sh, capture_inner_k23.sh
# and emergency_stop_k23.sh. Never run directly.
#
# Ownership is proven per PROCESS (k23_procs.py): every process of a batch inherits the environment token
# KCAP_OWNER="k23:<scene>:<stamp>" (capture_one_k23.sh itself runs with "k23o:<scene>:<invocation>"); sessions K23
# creates with `setsid` are verified by their tagged leader and then contain tokenless members too; every signal goes
# through a pidfd re-verified against the process start time. THIS FILE SIGNALS NOTHING: it only publishes and retires
# the records "pid boot_id starttime token" (thermal/{pgid,qcpgid,inner}_<stamp>), which are hints for a person and
# for the guard, never permission to signal (k23_procs.py record re-verifies them).
# The variable name deliberately does not start with K23_ (that prefix is reserved for the test overrides).

own_boot_id() { cat /proc/sys/kernel/random/boot_id 2>/dev/null; }
own_starttime() {   # $1 pid -> field 22 of /proc/<pid>/stat (clock ticks since boot), robust to spaces in comm
  local s; s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  s="${s##*) }"; set -- $s; echo "${20}"
}
own_publish() {     # $1 record file, $2 pid -> "pid boot_id starttime token" (atomic)
  local st; st=$(own_starttime "$2") || st=0
  printf '%s %s %s %s\n' "$2" "$(own_boot_id)" "${st:-0}" "${KCAP_OWNER:-}" > "$1.tmp" && mv -f "$1.tmp" "$1"
}
own_retire() {      # $1 record file -> moved to thermal/retired/ (kept as evidence, never signalled again)
  local d; d="$(dirname "$1")/retired"; mkdir -p "$d" && mv -f "$1" "$d/$(basename "$1").$(date +%s)" 2>/dev/null
}
