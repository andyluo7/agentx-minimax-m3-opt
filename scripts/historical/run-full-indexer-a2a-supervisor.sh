#!/usr/bin/env bash
set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly submit_script="$shared_root/submit-full-indexer-a2a.sh"
readonly recipe="$shared_root/repos/InferenceX/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp_full_indexer.sh"
readonly account=r7n
readonly partition=256C8G1H_MI355X_Ubuntu24
readonly reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
readonly node=vultr-mi355x-6
readonly sweep_id=full-indexer-a2a-20260825-r2
readonly recipe_sha256=76f025a44df07ff54ea4ceb6ec076f40403fa1a091e0ff1c03d4aa63b76a670d

exec 9>"$shared_root/logs/${sweep_id}.lock"
if ! flock -n 9; then
    printf '%s supervisor already active\n' "$(date -Is)"
    exit 1
fi

verify_staged_inputs() {
    local actual_sha256
    bash -n "$recipe" "$submit_script"
    actual_sha256="$(sha256sum "$recipe" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$recipe_sha256" ]]; then
        printf 'Recipe checksum mismatch: expected %s, got %s\n' \
            "$recipe_sha256" "$actual_sha256" >&2
        exit 2
    fi
    if grep -Eq -- '--hf-overrides|index_topk_freq|use_index_cache' "$recipe"; then
        printf 'Full-indexer recipe unexpectedly contains an index-cache override.\n' >&2
        exit 2
    fi
}

preflight_node() {
    local output status
    while true; do
        set +e
        output="$(
            srun \
                --account="$account" \
                --partition="$partition" \
                --reservation="$reservation" \
                --nodes=1 \
                --ntasks=1 \
                --exclusive \
                --gpus-per-node=8 \
                --nodelist="$node" \
                --time=00:05:00 \
                --immediate=30 \
                bash -lc '
                    set -euo pipefail
                    hostname
                    date -Is
                    rocm-smi --showuse --showmemuse --showpids
                    if kfd_pids="$(fuser /dev/kfd 2>/dev/null)"; then
                        printf "Unexpected /dev/kfd users: " >&2
                        printf "%s\n" "$kfd_pids" >&2
                        exit 21
                    fi
                    for used in /sys/class/drm/card*/device/mem_info_vram_used; do
                        [[ -r "$used" ]] || continue
                        value="$(<"$used")"
                        if ((value > 536870912)); then
                            printf "Unexpected VRAM use: %s=%s\n" "$used" "$value" >&2
                            exit 22
                        fi
                    done
                    ps -eo user,pid,ppid,stat,pcpu,pmem,etime,cmd --sort=-pcpu | sed -n "1,30p"
                    who || true
                    enroot list | grep -Fx minimaxm3-vllm-rocm-v0271
                    uptime
                    free -h
                    df -h / /tmp /shared/data
                    journalctl -k --since "2 hours ago" --no-pager 2>/dev/null \
                        | grep -Ei "amdgpu|xgmi|ras|gpu reset|page fault" \
                        | tail -80 || true
                ' 2>&1
        )"
        status=$?
        set -e
        printf '%s\n' "$output"
        if ((status == 0)); then
            return
        fi
        if grep -Eq 'QOSGrpSubmitJobsLimit|Requested nodes are busy|Unable to allocate resources' <<<"$output"; then
            printf '%s preflight could not allocate yet; retrying\n' "$(date -Is)"
            sleep 30
            continue
        fi
        printf '%s node preflight failed with status %s\n' "$(date -Is)" "$status" >&2
        exit "$status"
    done
}

submit_output=""
submit_when_available() {
    local point="$1"
    local c1_job_id="${2:-}"
    local status
    while true; do
        set +e
        if [[ "$point" == c1 ]]; then
            submit_output="$(
                FULL_INDEXER_POINT=c1 \
                SWEEP_ID_OVERRIDE="$sweep_id" \
                "$submit_script" 2>&1
            )"
        else
            submit_output="$(
                FULL_INDEXER_POINT=c32 \
                C1_JOB_ID="$c1_job_id" \
                SWEEP_ID_OVERRIDE="$sweep_id" \
                "$submit_script" 2>&1
            )"
        fi
        status=$?
        set -e
        printf '%s\n' "$submit_output"
        if ((status == 0)); then
            return
        fi
        if grep -Eq 'QOSGrpSubmitJobsLimit|Job violates accounting/QOS policy' \
                <<<"$submit_output"; then
            printf '%s submission slot was lost; retrying\n' "$(date -Is)"
            sleep 30
            continue
        fi
        printf '%s %s submission failed with status %s\n' \
            "$(date -Is)" "$point" "$status" >&2
        exit "$status"
    done
}

wait_for_job() {
    local job_id="$1"
    local state
    while squeue -h -j "$job_id" | grep -q .; do
        state="$(squeue -h -j "$job_id" -o '%T %M %R')"
        printf '%s job %s %s\n' "$(date -Is)" "$job_id" "$state"
        sleep 60
    done
    state="$(scontrol show job -o "$job_id" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')"
    printf '%s job %s final_state=%s\n' "$(date -Is)" "$job_id" "$state"
    [[ "$state" == COMPLETED ]]
}

verify_staged_inputs
preflight_node

submit_when_available c1
c1_output="$submit_output"
c1_job_id="$(awk -F= '$1 == "c1_resident_tp4" {print $2}' <<<"$c1_output")"
[[ "$c1_job_id" =~ ^[0-9]+$ ]]

if ! wait_for_job "$c1_job_id"; then
    printf '%s C1 failed; C32 will not be submitted.\n' "$(date -Is)" >&2
    exit 3
fi

submit_when_available c32 "$c1_job_id"
c32_output="$submit_output"
c32_job_id="$(awk -F'[= ]' '$1 == "c32_simple_tp4" {print $2}' <<<"$c32_output")"
[[ "$c32_job_id" =~ ^[0-9]+$ ]]

wait_for_job "$c32_job_id"
