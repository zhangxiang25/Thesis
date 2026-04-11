import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =================配置区域=================
# 这里填你刚刚生成的那个 csv 文件的路径
file_path = "outputs/4x4/custom_metrics_run1_ep50.csv"
# ==========================================

def plot_congestion_heatmap(csv_path):
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return

    # 1. 读取数据
    df = pd.read_csv(csv_path)

    # 2. 筛选出所有包含 "stop_ratio" 的列
    # 你的列名格式是: "AgentID_stop_ratio"
    ratio_cols = [c for c in df.columns if "stop_ratio" in c]

    if not ratio_cols:
        print("错误：CSV中没有找到包含 '_stop_ratio' 的列，请检查列名。")
        return

    # 3. [关键步骤] 对列名进行自然排序
    # 默认排序是: 0, 1, 10, 11... 2, 20...
    # 我们需要按 Agent ID 数字排序: 0, 1, 2, ..., 9, 10...
    # 逻辑：提取下划线前面的数字部分进行转换
    sorted_cols = sorted(ratio_cols, key=lambda x: int(x.split('_')[0]))

    # 4. 准备绘图数据
    # 转置矩阵 (Transpose)：让 行=Agent(路口), 列=Time(时间步)
    plot_data = df[sorted_cols].T
    
    # 修改索引名称，把 "0_stop_ratio" 简化为 "Agent 0" 以便图表好看
    plot_data.index = [x.split('_')[0] for x in plot_data.index]

    # 5. 绘制热力图
    plt.figure(figsize=(15, 8)) # 设置画布大小 (宽, 高)
    
    # 绘制 
    # cmap="RdYlGn_r": 颜色条。r表示reverse(反转)。
    # 绿色(0.0)代表畅通，红色(1.0)代表完全堵死
    sns.heatmap(data=plot_data, 
                cmap="RdYlGn_r", 
                vmin=0, vmax=1,  # 固定范围 0~1
                cbar_kws={'label': 'Stop Ratio (Congestion Level)'})

    plt.title(f"Traffic Congestion Heatmap\n(File: {os.path.basename(csv_path)})", fontsize=14)
    plt.xlabel("Simulation Step", fontsize=12)
    plt.ylabel("Intersection Agent ID", fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_congestion_heatmap(file_path)