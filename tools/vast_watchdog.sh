#!/bin/zsh
ID=$1; LOG=$2; MAX_MIN=${3:-180}; STALL_MIN=${4:-20}; IDLE_MIN=${5:-5}
OUT=~/.cache/photo2fcstd/watchdog_$ID.log
BW=/Users/pavliha/.local/bin/vastai
START=$(date +%s); UNREACH=0
say() { echo "$(date +%H:%M) $*" >> $OUT; }
kill_box() { say "DESTROY $ID: $1"; yes | $BW destroy instance $ID >> $OUT 2>&1; exit 0; }
say "watchdog start id=$ID log=$LOG max=${MAX_MIN}m stall=${STALL_MIN}m idle=${IDLE_MIN}m"
while true; do
  sleep 120
  INFO=$($BW show instances --raw 2>/dev/null | python3 -c "import json,sys; i=[x for x in json.load(sys.stdin) if x['id']==$ID]; print(i[0].get('ssh_host',''), i[0].get('ssh_port',''), i[0].get('actual_status','')) if i else print('gone')")
  [[ "$INFO" == "gone" ]] && { say "instance gone"; exit 0; }
  HOST=${INFO%% *}; REST=${INFO#* }; PORT=${REST%% *}
  ELAPSED=$(( ($(date +%s) - START) / 60 ))
  (( ELAPSED > MAX_MIN )) && kill_box "over time budget ${ELAPSED}m > ${MAX_MIN}m"
  R=$(ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes root@$HOST "test -f $LOG && echo \$(( (\$(date +%s) - \$(stat -c %Y $LOG)) / 60 )) \$(grep -c 'JOB_DONE' $LOG) || echo nolog" 2>/dev/null)
  if [[ -z "$R" ]]; then
    UNREACH=$((UNREACH + 1)); say "unreachable ($UNREACH)"
    (( UNREACH >= 7 )) && kill_box "unreachable for $((UNREACH * 2)) minutes"
    continue
  fi
  UNREACH=0
  [[ "$R" == "nolog" ]] && { (( ELAPSED > 30 )) && kill_box "no job log after ${ELAPSED}m"; say "no log yet (${ELAPSED}m)"; continue; }
  AGE=${R%% *}; DONE=${R#* }
  if (( DONE > 0 )); then
    (( AGE >= IDLE_MIN )) && kill_box "job done, idle ${AGE}m"
    say "done, idle ${AGE}m"; continue
  fi
  (( AGE >= STALL_MIN )) && kill_box "stalled: log untouched for ${AGE}m"
  say "alive: log ${AGE}m old, elapsed ${ELAPSED}m"
done
