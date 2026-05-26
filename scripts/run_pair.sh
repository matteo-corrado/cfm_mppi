#!/bin/bash -l
# scripts/discovery/run_pair.sh
#
# Pair-array sbatch: runs FP32 (task 0) + TF32 (task 1) for a single
# (dataset, dynamics) cell. Partition + gres are supplied via sbatch CLI
# override at submit time so the same script serves both L40S and H200.
#
# Usage (submitted by submit_sweep.sh, not directly):
#   sbatch --partition=gpuq --gres=gpu:l40s:1 --time=01:30:00 \
#          --job-name=cfm-ucy-uni-l40s-pair \
#          scripts/run_pair.sh ucy unicycle l40s

#SBATCH --account=free
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Thread caps — prevent compute libraries from spawning many threads on a node
# where we only own 4 cores (workload is GIL-bound + small JAX-CPU dispatches).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Diagnostic header — record CPU + NUMA + GPU state for env.json post-hoc.
# Boost is empirically active cluster-wide via governor=performance
# (probed 2026-05-26 per docs/00-foundation/discovery_hardware_specs.md).
# CRITICAL: lscpu CPU MHz is the static CPUID nominal value — only trust
# /proc/cpuinfo MHz and cpupower frequency-info current.
{
  echo "=== CPU state $(date -Iseconds) ==="
  echo "node: $(hostname)"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "intel_pstate no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo N/A)"
  echo "amd_boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo N/A)"
  echo "intel_pstate EPP: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo N/A)"
  echo "live MHz per core (first 4):"; grep MHz /proc/cpuinfo | head -4
  echo "scaling_max_freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo unknown)"
  cpupower frequency-info 2>/dev/null | head -10 || true
  echo "numa layout:"; numactl --hardware 2>/dev/null | head -5 || echo "N/A"
  echo "slurm cpus_per_task: ${SLURM_CPUS_PER_TASK:-N/A}"
  echo "slurm cpu_bind: ${SLURM_CPU_BIND:-N/A}"
} >&2

ds="${1:?dataset required: sfm|ucy|sdd}"
dyn="${2:?dynamics required: unicycle|doubleintegrator}"
gpu="${3:-l40s}"

PRECS=(fp32 tf32)
prec="${PRECS[$SLURM_ARRAY_TASK_ID]}"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate cfm_mppi

# Preflight
[[ -f ./output_dir/cfm_transformer/checkpoint.pth ]] || { echo "FATAL: missing checkpoint" >&2; exit 2; }
[[ -f ./output_dir/cfm_transformer/args.json ]] || { echo "FATAL: missing args.json" >&2; exit 2; }
if [[ "$ds" != "sfm" ]]; then
    [[ -f "./dataset/eval80_obs_${ds}.pkl" ]] || { echo "FATAL: missing eval80_obs_${ds}.pkl" >&2; exit 2; }
    [[ -f "./dataset/eval80_ego_${ds}.pt"  ]] || { echo "FATAL: missing eval80_ego_${ds}.pt" >&2; exit 2; }
fi

# Env sanity — JAX must be CPU-only (no GPU jaxlib contending with torch)
python -c "import torch; assert torch.cuda.is_available()" || { echo "FATAL: cuda not available" >&2; exit 3; }
python -c "import jax; assert all(d.platform=='cpu' for d in jax.devices()), jax.devices()" || { echo "FATAL: jax not CPU-only" >&2; exit 3; }

# Cell + per-GPU result root
case "$dyn" in
    unicycle)         dyn_short="uni" ;;
    doubleintegrator) dyn_short="db"  ;;
    *) echo "FATAL: unknown dyn: $dyn" >&2; exit 4 ;;
esac
cell="${ds}_${dyn_short}_${prec}"
result_root="results_${gpu}"

# Idempotency guard — per-GPU isolation, so L40S+H200 of same cell don't collide
if [[ -f "${result_root}/${cell}/cfm_mppi.txt" ]]; then
    echo "SKIP: ${result_root}/${cell} already complete"
    exit 0
fi

mkdir -p logs "${result_root}/${cell}"

echo "=== ${cell} starting $(date -Iseconds) on $(hostname) gpu=${gpu} ==="
nvidia-smi -L

export EVAL_RESULT_ROOT="${result_root}"
python -u "cfm_mppi/evaluation/eval_cfm_mppi_${dyn}.py" "$ds" "$prec"
rc=$?

nvidia-smi
echo "=== ${cell} finished $(date -Iseconds) rc=${rc} ==="
exit "$rc"
