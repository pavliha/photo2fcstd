#!/bin/bash
LOG=$1; MAX_SEC=${2:-10800}; shift 2
{ echo "JOB_START $(date +%H:%M) $*"; timeout $MAX_SEC "$@"; echo "JOB_EXIT $? $(date +%H:%M)"; echo JOB_DONE; } >> $LOG 2>&1
