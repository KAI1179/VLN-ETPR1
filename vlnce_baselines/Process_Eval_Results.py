import os
import json
import glob
import re
from typing import List, Dict, Set

# 尝试导入matplotlib，如果失败则提示安装
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("错误：缺少 matplotlib 库。请使用 'pip install matplotlib' 命令安装。")
    exit()


def find_and_parse_json_files(
    results_dir: str, metric1: str, metric2: str
) -> List[Dict]:
    """
    在指定目录中查找并解析所有符合命名规范的JSON实验结果文件。
    此函数现在是通用的，可以解析任何指定的两个指标。

    Args:
        results_dir: 存放JSON文件的文件夹路径。
        metric1: 第一个需要提取的指标名称 (e.g., 'success' or 'sdtw').
        metric2: 第二个需要提取的指标名称 (e.g., 'spl' or 'ndtw').

    Returns:
        一个列表，其中每个元素都是一个包含step和各项指标的字典。
    """
    json_pattern = os.path.join(results_dir, "stats_ckpt_*_val_unseen.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(
            f"警告：在文件夹 '{results_dir}' 中没有找到任何匹配 'stats_ckpt_*_val_unseen.json' 格式的文件。"
        )
        return []

    all_results = []
    print(f"找到了 {len(json_files)} 个结果文件，正在解析...")

    # 动态生成组合指标的名称
    composite_metric_name = f"{metric2}_plus_{metric1}"

    for file_path in json_files:
        match = re.search(
            r"stats_ckpt_(\d+)_val_unseen\.json", os.path.basename(file_path)
        )
        if not match:
            continue

        step = int(match.group(1))

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

                # 使用传入的参数来检查和提取数据
                if metric1 in data and metric2 in data:
                    result_dict = {
                        "step": step,
                        metric1: data[metric1],
                        metric2: data[metric2],
                        composite_metric_name: data[metric1] + data[metric2],
                    }
                    all_results.append(result_dict)
                else:
                    print(
                        f"警告：文件 '{file_path}' 缺少 '{metric1}' 或 '{metric2}' 键。"
                    )

        except json.JSONDecodeError:
            print(f"警告：无法解析文件 '{file_path}'，可能不是一个有效的JSON文件。")
        except Exception as e:
            print(f"处理文件 '{file_path}' 时发生未知错误: {e}")

    all_results.sort(key=lambda x: x["step"])
    return all_results


def print_top_n_ranking(
    results: List[Dict],
    ranking_metric: str,
    display_metric1: str,
    display_metric2: str,
    n: int = 5,
):
    """
    根据指定指标对结果进行排名并打印前n名。
    此函数现在可以显示任何指定的两个指标作为列。

    Args:
        results: 包含所有实验结果的列表。
        ranking_metric: 用于排名的指标名称 (e.g., 'success', 'sdtw', 'spl_plus_success').
        display_metric1: 表格中要显示的第一列指标名称。
        display_metric2: 表格中要显示的第二列指标名称。
        n: 打印排名前几位。
    """
    if not results:
        return

    sorted_results = sorted(results, key=lambda x: x[ranking_metric], reverse=True)

    metric_display_name = ranking_metric.replace("_", " ").upper()

    # 动态生成表头
    header1 = display_metric1.upper()
    header2 = display_metric2.upper()

    print("\n" + "=" * 60)
    print(f"🏆 按 {metric_display_name} 指标排名的前 {n} 名 🏆")
    print("=" * 60)
    print(f"{'排名':<5}{'Step':<10}{header1:<20}{header2:<20}")
    print("-" * 60)

    for i, res in enumerate(sorted_results[:n]):
        rank = i + 1
        step = res["step"]

        # 根据指标类型选择不同的格式化方式
        if display_metric1 == "success":
            metric1_str = f"{res[display_metric1]:.2%}"
        else:
            metric1_str = f"{res[display_metric1]:.4f}"

        metric2_str = f"{res[display_metric2]:.4f}"

        print(f"{rank:<5}{step:<10}{metric1_str:<20}{metric2_str:<20}")
    print("=" * 60)


def plot_metric_over_steps(results: List[Dict], metric: str, save_dir):
    """
    绘制指定指标随step变化的折线图。(此函数无需修改，本身就是通用的)
    """
    if not results or len(results) < 2:
        print("\n信息：结果数量不足，无法生成趋势图。")
        return

    steps = [res["step"] for res in results]
    metric_values = [res[metric] for res in results]

    metric_display_name = metric.replace("_", " ").upper()

    plt.figure(figsize=(12, 7))
    plt.plot(steps, metric_values, marker="o", linestyle="-", label=metric_display_name)
    plt.title(f"{metric_display_name} Over Steps", fontsize=16)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel(metric_display_name, fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{metric}_over_steps.png")
    plt.savefig(save_path)
    print(f"\n📈 趋势图已保存至: {save_path}")


def cleanup_checkpoints(checkpoints_dir: str, top_steps: Set[int]):
    """
    清理两个排名前五之外的checkpoints。(此函数无需修改，本身就是通用的)
    """
    ckpt_pattern = os.path.join(checkpoints_dir, "ckpt.iter*.pth")
    all_ckpts = glob.glob(ckpt_pattern)

    if not all_ckpts:
        print(
            f"\n信息：在文件夹 '{checkpoints_dir}' 中没有找到任何匹配 'ckpt.iter*.pth' 格式的 checkpoint 文件。"
        )
        return

    checkpoints_to_delete = []
    for ckpt_path in all_ckpts:
        match = re.search(r"ckpt\.iter(\d+)\.pth", os.path.basename(ckpt_path))
        if match:
            step = int(match.group(1))
            if step not in top_steps:
                checkpoints_to_delete.append(ckpt_path)

    if not checkpoints_to_delete:
        print("\n🎉 所有现存的 checkpoint 都在排名的前列，无需清理。")
        return

    print("\n" + "=" * 50)
    print("🗑️ Checkpoint 清理向导")
    print("=" * 50)
    print(f"共发现 {len(all_ckpts)} 个 checkpoints。")
    print(f"其中有 {len(checkpoints_to_delete)} 个不在任一指标的前五名中，可以被删除。")
    print(f"将被保留的 Steps: {sorted(list(top_steps))}")

    try:
        confirm = input(
            f"\n⚠️ 是否要删除这 {len(checkpoints_to_delete)} 个 checkpoint 文件？这是一个不可逆操作！(输入 'yes' 确认): "
        )
        if confirm.lower() == "yes":
            second_confirm = input(
                f"\n⚠️ 再次确认删除！这是一个不可逆操作！(输入 'yes' 确认): "
            )
            if second_confirm.lower() == "yes":
                print("\n正在删除文件...")
                deleted_count = 0
                for file_path in checkpoints_to_delete:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except OSError as e:
                        print(f"删除文件 '{file_path}' 失败: {e}")
                print(f"\n✅ 操作完成！成功删除了 {deleted_count} 个 checkpoint 文件。")
            else:
                print("\n❌ 操作已取消，没有文件被删除。")
        else:
            print("\n❌ 操作已取消，没有文件被删除。")
    except KeyboardInterrupt:
        print("\n\n操作被用户中断，没有文件被删除。")


def main():
    # --- 新增：获取用户选择 ---
    task_choice = ""
    while task_choice not in ["r2r", "rxr"]:
        task_choice = input("请输入要分析的任务类型 (r2r / rxr): ").lower().strip()
        if task_choice not in ["r2r", "rxr"]:
            print("输入无效，请输入 'r2r' 或 'rxr'。")

    # --- 新增：根据用户选择定义指标 ---
    if task_choice == "r2r":
        primary_metric = "success"
        secondary_metric = "spl"
        print("\n已选择 R2R 任务，将使用 Success 和 SPL 指标进行分析。")
    else:  # rxr
        primary_metric = "sdtw"
        secondary_metric = "ndtw"
        print("\n已选择 RxR 任务，将使用 SDTW 和 NDTW 指标进行分析。")

    composite_metric = f"{secondary_metric}_plus_{primary_metric}"

    # 这部分路径你可以根据需要修改回原来的固定路径
    # base_dir = input("请输入 checkpoint 的根目录 (例如: data/logs/checkpoints/release_r2r_dagger): ")
    base_dir = "data/logs/checkpoints/release_r2r_dagger"
    results_dir = os.path.join(base_dir, "store", "eval_results")
    checkpoints_dir = base_dir

    if not os.path.isdir(results_dir):
        print(f"错误: 结果文件夹 '{results_dir}' 不存在。请检查路径。")
        return
    if not os.path.isdir(checkpoints_dir):
        print(f"错误: checkpoints 文件夹 '{checkpoints_dir}' 不存在。请检查路径。")
        return

    # 1. 读取并解析所有JSON文件
    all_results = find_and_parse_json_files(
        results_dir, primary_metric, secondary_metric
    )
    if not all_results:
        print("没有找到有效数据，程序退出。")
        return

    # 2. 按主要指标排名并打印
    print_top_n_ranking(all_results, primary_metric, primary_metric, secondary_metric)

    # 3. 按组合指标排名并打印
    print_top_n_ranking(all_results, composite_metric, primary_metric, secondary_metric)

    # 4. 绘制组合指标变化图
    plot_metric_over_steps(all_results, composite_metric, checkpoints_dir)

    # 5. 准备清理Checkpoints
    top_primary_steps = {
        res["step"]
        for res in sorted(all_results, key=lambda x: x[primary_metric], reverse=True)[
            :5
        ]
    }
    top_composite_steps = {
        res["step"]
        for res in sorted(all_results, key=lambda x: x[composite_metric], reverse=True)[
            :5
        ]
    }

    all_top_steps = top_primary_steps.union(top_composite_steps)

    # 6. 执行清理操作
    cleanup_checkpoints(checkpoints_dir, all_top_steps)


if __name__ == "__main__":
    main()
