#!/bin/bash
# Hardware monitor: samples GPU and CPU stats every INTERVAL seconds.
# Usage: monitor_hw.sh <output_csv> [interval_seconds]
# Writes CSV rows: timestamp,gpu_util,gpu_mem_used,gpu_mem_total,cpu_util,ram_used,ram_total

OUTPUT="$1"
INTERVAL="${2:-30}" # Poll every 30 seconds

echo "timestamp,gpu_util_%,gpu_mem_used_MiB,gpu_mem_total_MiB,cpu_util_%,ram_used_MiB,ram_total_MiB" > "$OUTPUT"

get_cpu_util() {
    # Read /proc/stat twice, 1s apart, compute utilisation
    read -r _ u1 n1 s1 i1 _ < /proc/stat
    sleep 1
    read -r _ u2 n2 s2 i2 _ < /proc/stat
    used=$(( (u2-u1) + (n2-n1) + (s2-s1) ))
    total=$(( used + (i2-i1) ))
    if [ "$total" -eq 0 ]; then echo "0"; else echo $(( used * 100 / total )); fi
}

get_ram_mib() {
    total=$(awk '/^MemTotal:/ {print int($2/1024)}' /proc/meminfo)
    avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    used=$(( total - avail ))
    echo "$used $total"
}

while true; do
    TS=$(date +%Y-%m-%dT%H:%M:%S)

    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
          --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU_UTIL=$(echo "$GPU" | cut -d',' -f1)
    GPU_MEM_USED=$(echo "$GPU" | cut -d',' -f2)
    GPU_MEM_TOTAL=$(echo "$GPU" | cut -d',' -f3)

    CPU_UTIL=$(get_cpu_util)
    read -r RAM_USED RAM_TOTAL <<< "$(get_ram_mib)"

    echo "$TS,$GPU_UTIL,$GPU_MEM_USED,$GPU_MEM_TOTAL,$CPU_UTIL,$RAM_USED,$RAM_TOTAL" >> "$OUTPUT"

    sleep $(( INTERVAL - 1 ))   # -1 to account for the 1s cpu measurement
done
