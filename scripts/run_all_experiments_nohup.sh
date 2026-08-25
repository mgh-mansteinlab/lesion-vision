#!/usr/bin/env bash
# Run every configs/experiments/*.yaml sequentially (one job at a time — avoids GPU fights).
#
# Usage:
#   export DATA_DIR=/path/to/tiles
#   export OUTPUT_DIR=./models          # optional
#   export WORLD_SIZE=1                 # or 8 for multi-GPU
#   nohup bash scripts/run_all_experiments_nohup.sh > experiments/logs/nohup_master.log 2>&1 &
#
# Env:
#   SKIP_TEMPLATES=1   (default) skip YAMLs with "template" in the filename
#   SKIP_TEMPLATES=0   run those too
#   YAML_GLOB=*.yaml   or e.g. baseline*.yaml

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_DIR}"

LOG_ROOT="${PROJECT_DIR}/experiments/logs"
mkdir -p "${LOG_ROOT}"

MASTER_STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_ROOT}/nohup_master_${MASTER_STAMP}.log"
SUMMARY_TSV="${LOG_ROOT}/run_summary.tsv"
JOURNAL_MD="${LOG_ROOT}/RUN_JOURNAL.md"
APPEND_TSV="${PROJECT_DIR}/experiments/EXPERIMENT_LOG_APPEND.tsv"
SKIP_TEMPLATES="${SKIP_TEMPLATES:-1}"
YAML_GLOB="${YAML_GLOB:-*.yaml}"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========================================"
echo "run_all_experiments_nohup started $(date -Iseconds)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_DIR=${DATA_DIR:-<unset>}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/models}"
echo "WORLD_SIZE=${WORLD_SIZE:-1}"
echo "SKIP_TEMPLATES=${SKIP_TEMPLATES}"
echo "YAML_GLOB=${YAML_GLOB}"
echo "Master log: ${MASTER_LOG}"
echo "========================================"

if [[ -z "${DATA_DIR:-}" ]]; then
  echo "WARNING: DATA_DIR is unset — training will likely fail until you export DATA_DIR."
fi

if [[ ! -f "${SUMMARY_TSV}" ]] || ! grep -q '^timestamp_iso' "${SUMMARY_TSV}" 2>/dev/null; then
  echo -e "timestamp_iso\tyaml_rel\texit_code\tduration_sec\tlog_file" > "${SUMMARY_TSV}"
fi

if [[ ! -s "${JOURNAL_MD}" ]]; then
  {
    echo "# Experiment run journal (auto-generated)"
    echo ""
    echo "Per-run details below. Machine-readable summary: \`experiments/logs/run_summary.tsv\`."
    echo ""
  } > "${JOURNAL_MD}"
fi

if [[ ! -f "${APPEND_TSV}" ]] || ! grep -q '^date' "${APPEND_TSV}" 2>/dev/null; then
  echo -e "date\texperiment_id\tconfig_relpath\texit_code\tduration_s\tlog_relpath" > "${APPEND_TSV}"
fi

shopt -s nullglob
mapfile -t YAMLS < <(
  find "${PROJECT_DIR}/configs/experiments" -maxdepth 1 -type f -name "${YAML_GLOB}" | LC_ALL=C sort
)

if [[ ${#YAMLS[@]} -eq 0 ]]; then
  echo "No YAML files matched configs/experiments/${YAML_GLOB}"
  exit 1
fi

echo "Planned runs: ${#YAMLS[@]} file(s)"

for yaml in "${YAMLS[@]}"; do
  base="$(basename "${yaml}" .yaml)"
  if [[ "${SKIP_TEMPLATES}" == "1" ]] && [[ "${base}" == *template* ]]; then
    echo "[skip] ${yaml} (template — set SKIP_TEMPLATES=0 to include)"
    continue
  fi

  stamp="$(date +%Y%m%d_%H%M%S)"
  run_log="${LOG_ROOT}/${base}_${stamp}.log"
  rel_yaml="${yaml#${PROJECT_DIR}/}"
  rel_log="experiments/logs/$(basename "${run_log}")"

  echo ""
  echo "---- $(date -Iseconds) START ${rel_yaml} ----"

  start_ts=$(date +%s)
  set +e
  env PYTHONUNBUFFERED=1 DATA_DIR="${DATA_DIR:-}" OUTPUT_DIR="${OUTPUT_DIR:-}" WORLD_SIZE="${WORLD_SIZE:-1}" \
    python3 "${SCRIPT_DIR}/launch_experiment.py" "${yaml}" >"${run_log}" 2>&1
  code=$?
  set -e
  end_ts=$(date +%s)
  dur=$((end_ts - start_ts))

  ts_iso="$(date -Iseconds)"
  printf '%s\t%s\t%s\t%s\t%s\n' "${ts_iso}" "${rel_yaml}" "${code}" "${dur}" "${rel_log}" >> "${SUMMARY_TSV}"

  {
    echo ""
    echo "## ${ts_iso} — \`${base}\`"
    echo "- **Config:** \`${rel_yaml}\`"
    echo "- **Exit code:** ${code}"
    echo "- **Duration (s):** ${dur}"
    echo "- **Log:** \`${rel_log}\`"
    echo ""
  } >> "${JOURNAL_MD}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%Y-%m-%d)" "${base}" "${rel_yaml}" "${code}" "${dur}" "${rel_log}" >> "${APPEND_TSV}"

  echo "---- END ${base} exit=${code} duration=${dur}s log=${rel_log} ----"
done

echo ""
echo "========================================"
echo "All queued runs finished $(date -Iseconds)"
echo "Summary TSV: ${SUMMARY_TSV}"
echo "Journal: ${JOURNAL_MD}"
echo "Paper-oriented TSV rows: ${APPEND_TSV}"
echo "========================================"
