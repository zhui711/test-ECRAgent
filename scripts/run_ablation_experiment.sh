#!/bin/bash
#
# 消融实验运行脚本
# ====================
#
# 提供三种消融模式：
# 1. 完整消融 (Full Ablation): 消融 Golden Graph + Memory Bank
# 2. 仅消融 Golden Graph
# 3. 仅消融 Memory Bank
#
# Usage:
#   bash scripts/run_ablation_experiment.sh [mode] [options]
#
#   mode:
#     full       - 完整消融 (--no-golden-graph --no-memory)
#     no-gg      - 仅消融 Golden Graph (--no-golden-graph)
#     no-memory  - 仅消融 Memory Bank (--no-memory)
#     baseline   - 基线模式 (使用所有 Offline 组件)
#
# Examples:
#   bash scripts/run_ablation_experiment.sh full --limit 10
#   bash scripts/run_ablation_experiment.sh no-gg --output-dir output_ablation_gg
#   bash scripts/run_ablation_experiment.sh baseline --limit 5
#

set -e

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 默认参数
MODE="${1:-full}"
shift || true

# 根据模式设置消融参数
case "$MODE" in
    full)
        echo "=========================================="
        echo "Running FULL ABLATION Experiment"
        echo "  - Golden Graph: DISABLED"
        echo "  - Memory Bank: DISABLED"
        echo "=========================================="
        ABLATION_FLAGS="--no-golden-graph --no-memory"
        OUTPUT_DIR="output_ablation_full"
        ;;
    no-gg|no-golden-graph)
        echo "=========================================="
        echo "Running GOLDEN GRAPH ABLATION Experiment"
        echo "  - Golden Graph: DISABLED"
        echo "  - Memory Bank: ENABLED"
        echo "=========================================="
        ABLATION_FLAGS="--no-golden-graph"
        OUTPUT_DIR="output_ablation_no_gg"
        ;;
    no-memory|no-mem)
        echo "=========================================="
        echo "Running MEMORY BANK ABLATION Experiment"
        echo "  - Golden Graph: ENABLED"
        echo "  - Memory Bank: DISABLED"
        echo "=========================================="
        ABLATION_FLAGS="--no-memory"
        OUTPUT_DIR="output_ablation_no_memory"
        ;;
    baseline|full-model)
        echo "=========================================="
        echo "Running BASELINE (Full Model)"
        echo "  - Golden Graph: ENABLED"
        echo "  - Memory Bank: ENABLED"
        echo "=========================================="
        ABLATION_FLAGS=""
        OUTPUT_DIR="output_baseline"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [full|no-gg|no-memory|baseline] [additional options]"
        exit 1
        ;;
esac

# 检查是否指定了自定义输出目录
CUSTOM_OUTPUT=""
for arg in "$@"; do
    if [[ "$arg" == --output-dir=* ]]; then
        CUSTOM_OUTPUT="${arg#*=}"
    elif [[ "$arg" == --output-dir ]]; then
        CUSTOM_OUTPUT="NEXT"
    elif [[ "$CUSTOM_OUTPUT" == "NEXT" ]]; then
        CUSTOM_OUTPUT="$arg"
    fi
done

if [[ -n "$CUSTOM_OUTPUT" && "$CUSTOM_OUTPUT" != "NEXT" ]]; then
    OUTPUT_DIR="$CUSTOM_OUTPUT"
fi

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# 运行实验
python scripts/run_online_inference.py \
    $ABLATION_FLAGS \
    --output-dir "$OUTPUT_DIR" \
    --golden-graph-dir golden_graphs_refined_1 \
    "$@"

echo ""
echo "=========================================="
echo "Ablation experiment complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="



