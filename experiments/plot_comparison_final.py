import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# ================= 配置区域 =================
# 1. Q-Learning 文件路径模板 (根据你的截图 image_7815ba.png)
# 注意：你的QL文件在 outputs/4x4/ 下
QL_TEMPLATE = "outputs/4x4/ql-4x4grid_run1_conn0_ep{}.csv"

# 2. PPO 测试文件路径模板 (你刚刚生成的)
# 注意：你的PPO文件在 outputs/4x4grid/ 下，且前缀是 ppo_test_final
PPO_TEMPLATE = "outputs/4x4grid/ppo_test_final_conn0_ep{}.csv"

# 3. 对比参数
MAX_EPISODES = 20  # 刚才跑了20局
METRIC_COL = "system_total_waiting_time" # 核心指标
METRIC_NAME = "Total Waiting Time (s)"
# ===========================================

def get_episode_metrics(template, label):
    episodes = []
    values = []
    
    print(f"--- 正在读取 {label} 数据 ---")
    for i in range(1, MAX_EPISODES + 1):
        file_path = template.format(i)
        
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到文件: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            # 计算这一局的平均等待时间
            avg_value = df[METRIC_COL].mean()
            
            episodes.append(i)
            values.append(avg_value)
            print(f"  Episode {i}: {avg_value:.2f}")
        except Exception as e:
            print(f"❌ 读取错误 {file_path}: {e}")

    return pd.DataFrame({"Episode": episodes, "Value": values, "Algorithm": label})

def plot_final_comparison():
    # 1. 读取数据
    df_ql = get_episode_metrics(QL_TEMPLATE, "Q-Learning")
    df_ppo = get_episode_metrics(PPO_TEMPLATE, "PPO")
    
    # 合并数据
    df_all = pd.concat([df_ql, df_ppo])
    
    if df_all.empty:
        print("❌ 错误：没有读到任何数据，请检查路径配置！")
        return

    # 2. 开始绘图
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    # 画线图
    sns.lineplot(
        data=df_all, 
        x="Episode", 
        y="Value", 
        hue="Algorithm", 
        marker="o", 
        linewidth=2.5
    )

    plt.title(f"Performance Comparison: Q-Learning vs PPO", fontsize=16)
    plt.ylabel(METRIC_NAME, fontsize=12)
    plt.xlabel("Episode", fontsize=12)
    plt.xticks(range(1, MAX_EPISODES + 1)) # 显示所有刻度
    
    # 保存图片
    save_name = "final_comparison_result.png"
    plt.savefig(save_name, dpi=300)
    print(f"\n✅ 对比图已保存为: {save_name}")
    plt.show()

if __name__ == "__main__":
    plot_final_comparison()