import re
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class TrainingLog:
    """训练日志数据类"""
    steps: List[int]
    val_losses: List[float]
    train_times: List[float]
    step_avgs: List[float]
    name: str

def parse_log_file(filepath: str, name: str) -> TrainingLog:
    """
    解析训练日志文件，提取关键指标
    
    日志格式示例：
    step:0/2225 val_loss:10.8258 train_time:0ms step_avg:0.02ms
    step:250/2225 val_loss:5.7943 train_time:25095ms step_avg:100.38ms
    """
    steps = []
    val_losses = []
    train_times = []
    step_avgs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 只处理包含val_loss的行（验证步骤）
            if 'val_loss:' in line:
                # 提取step
                step_match = re.search(r'step:(\d+)/', line)
                if step_match:
                    step = int(step_match.group(1))
                    steps.append(step)
                
                # 提取val_loss
                val_loss_match = re.search(r'val_loss:(\d+\.\d+)', line)
                if val_loss_match:
                    val_loss = float(val_loss_match.group(1))
                    val_losses.append(val_loss)
                
                # 提取train_time (以秒为单位)
                time_match = re.search(r'train_time:(\d+)ms', line)
                if time_match:
                    train_time_ms = int(time_match.group(1))
                    train_times.append(train_time_ms / 1000.0)  # 转换为秒
                
                # 提取step_avg
                avg_match = re.search(r'step_avg:(\d+\.\d+)ms', line)
                if avg_match:
                    step_avg = float(avg_match.group(1))
                    step_avgs.append(step_avg / 1000.0)  # 转换为秒
    
    return TrainingLog(
        steps=steps,
        val_losses=val_losses,
        train_times=train_times,
        step_avgs=step_avgs,
        name=name
    )

def create_comparison_plot(log1: TrainingLog, log2: TrainingLog, 
                          output_path: str = "val_loss_comparison.png"):
    """
    创建验证损失对比图，保持与原始代码一致的风格
    """
    # 设置matplotlib风格，保持简洁专业
    plt.style.use('default')
    
    # 创建图形和子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    # 设置颜色
    colors = ['#1f77b4', '#ff7f0e']  # 与matplotlib默认颜色一致
    
    # 1. 验证损失对比 (主要指标)
    ax1 = axes[0, 0]
    ax1.plot(log1.steps, log1.val_losses, 
             label=log1.name, color=colors[0], linewidth=2, marker='o', markersize=4)
    ax1.plot(log2.steps, log2.val_losses, 
             label=log2.name, color=colors[1], linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Training Step', fontsize=11)
    ax1.set_ylabel('Validation Loss', fontsize=11)
    ax1.set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_yscale('log')  # 对数刻度，因为损失值范围较大
    
    # 添加最佳损失标记
    min_loss1 = min(log1.val_losses)
    min_loss2 = min(log2.val_losses)
    min_step1 = log1.steps[log1.val_losses.index(min_loss1)]
    min_step2 = log2.steps[log2.val_losses.index(min_loss2)]
    
    ax1.scatter(min_step1, min_loss1, color=colors[0], s=100, zorder=5,
                edgecolors='black', linewidth=1.5)
    ax1.scatter(min_step2, min_loss2, color=colors[1], s=100, zorder=5,
                edgecolors='black', linewidth=1.5)
    ax1.text(min_step1, min_loss1*1.1, f'{min_loss1:.3f}', 
             ha='center', fontsize=9, fontweight='bold')
    ax1.text(min_step2, min_loss2*1.1, f'{min_loss2:.3f}', 
             ha='center', fontsize=9, fontweight='bold')
    
    # 2. 累计训练时间对比
    ax2 = axes[0, 1]
    ax2.plot(log1.steps, log1.train_times, 
             label=f'{log1.name} (final: {log1.train_times[-1]:.0f}s)', 
             color=colors[0], linewidth=2, alpha=0.8)
    ax2.plot(log2.steps, log2.train_times, 
             label=f'{log2.name} (final: {log2.train_times[-1]:.0f}s)', 
             color=colors[1], linewidth=2, alpha=0.8)
    ax2.set_xlabel('Training Step', fontsize=11)
    ax2.set_ylabel('Cumulative Training Time (s)', fontsize=11)
    ax2.set_title('Training Time Accumulation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    
    # 3. 每步平均时间对比
    ax3 = axes[1, 0]
    ax3.plot(log1.steps, log1.step_avgs, 
             label=f'{log1.name} (avg: {np.mean(log1.step_avgs):.3f}s)', 
             color=colors[0], linewidth=2, alpha=0.7)
    ax3.plot(log2.steps, log2.step_avgs, 
             label=f'{log2.name} (avg: {np.mean(log2.step_avgs):.3f}s)', 
             color=colors[1], linewidth=2, alpha=0.7)
    ax3.set_xlabel('Training Step', fontsize=11)
    ax3.set_ylabel('Average Step Time (s)', fontsize=11)
    ax3.set_title('Per-Step Training Time', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    
    # 4. 损失下降率对比（一阶差分）
    ax4 = axes[1, 1]
    
    # 计算损失下降率
    if len(log1.val_losses) > 1 and len(log2.val_losses) > 1:
        loss_diff1 = np.diff(log1.val_losses) / np.diff(log1.steps)
        loss_diff2 = np.diff(log2.val_losses) / np.diff(log2.steps)
        
        # 使用移动平均平滑
        window_size = min(5, len(loss_diff1), len(loss_diff2))
        smooth_diff1 = np.convolve(loss_diff1, np.ones(window_size)/window_size, mode='valid')
        smooth_diff2 = np.convolve(loss_diff2, np.ones(window_size)/window_size, mode='valid')
        
        steps1 = log1.steps[1:len(smooth_diff1)+1]
        steps2 = log2.steps[1:len(smooth_diff2)+1]
        
        ax4.plot(steps1, smooth_diff1, 
                 label=f'{log1.name} (avg: {np.mean(smooth_diff1):.3e}/step)', 
                 color=colors[0], linewidth=2, alpha=0.7)
        ax4.plot(steps2, smooth_diff2, 
                 label=f'{log2.name} (avg: {np.mean(smooth_diff2):.3e}/step)', 
                 color=colors[1], linewidth=2, alpha=0.7)
        ax4.set_xlabel('Training Step', fontsize=11)
        ax4.set_ylabel('Loss Reduction Rate', fontsize=11)
        ax4.set_title('Loss Reduction Rate (Smoothed)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(frameon=True, fancybox=True, shadow=True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"对比图已保存至: {output_path}")

def print_summary_stats(log1: TrainingLog, log2: TrainingLog):
    """打印统计摘要"""
    print("=" * 60)
    print("TRAINING PERFORMANCE SUMMARY")
    print("=" * 60)
    
    def print_log_stats(log: TrainingLog):
        print(f"\n{log.name}:")
        print(f"  Steps analyzed: {len(log.steps)}")
        print(f"  Initial loss: {log.val_losses[0]:.4f}")
        print(f"  Final loss: {log.val_losses[-1]:.4f}")
        print(f"  Best loss: {min(log.val_losses):.4f} at step {log.steps[log.val_losses.index(min(log.val_losses))]}")
        print(f"  Total training time: {log.train_times[-1]:.2f}s ({log.train_times[-1]/60:.2f}min)")
        print(f"  Average step time: {np.mean(log.step_avgs):.3f}s")
        if len(log.val_losses) > 1:
            improvement = ((log.val_losses[0] - log.val_losses[-1]) / log.val_losses[0]) * 100
            print(f"  Improvement: {improvement:.1f}% reduction")
    
    print_log_stats(log1)
    print_log_stats(log2)
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS:")
    print("=" * 60)
    
    # 比较最终损失
    if log1.val_losses[-1] < log2.val_losses[-1]:
        diff = log2.val_losses[-1] - log1.val_losses[-1]
        perc = (diff / log2.val_losses[-1]) * 100
        print(f"✓ {log1.name} has lower final loss by {diff:.4f} ({perc:.1f}% better)")
    else:
        diff = log1.val_losses[-1] - log2.val_losses[-1]
        perc = (diff / log1.val_losses[-1]) * 100
        print(f"✓ {log2.name} has lower final loss by {diff:.4f} ({perc:.1f}% better)")
    
    # 比较训练效率
    efficiency1 = (log1.val_losses[0] - log1.val_losses[-1]) / log1.train_times[-1]
    efficiency2 = (log2.val_losses[0] - log2.val_losses[-1]) / log2.train_times[-1]
    
    if efficiency1 > efficiency2:
        ratio = efficiency1 / efficiency2
        print(f"✓ {log1.name} is {ratio:.2f}x more efficient (loss reduction per second)")
    else:
        ratio = efficiency2 / efficiency1
        print(f"✓ {log2.name} is {ratio:.2f}x more efficient (loss reduction per second)")
    
    # 比较步长时间
    if np.mean(log1.step_avgs) < np.mean(log2.step_avgs):
        ratio = np.mean(log2.step_avgs) / np.mean(log1.step_avgs)
        print(f"✓ {log1.name} steps are {ratio:.2f}x faster on average")
    else:
        ratio = np.mean(log1.step_avgs) / np.mean(log2.step_avgs)
        print(f"✓ {log2.name} steps are {ratio:.2f}x faster on average")

def main():
    """
    主函数：读取两个日志文件并生成对比图
    """
    # 配置参数
    FILE1_PATH = "/data1/sunqi/yue/modded-nanogpt/logs/adamw.txt"  # 请替换为你的第一个日志文件路径
    FILE2_PATH = "/data1/sunqi/yue/modded-nanogpt/logs/muon.txt"  # 请替换为你的第二个日志文件路径
    OUTPUT_PATH = "training_comparison.png"
    
    # 给每个实验命名
    EXP1_NAME = "AdamW"
    EXP2_NAME = "Muon"
    
    try:
        # 解析日志文件
        print("Parsing log files...")
        log1 = parse_log_file(FILE1_PATH, EXP1_NAME)
        log2 = parse_log_file(FILE2_PATH, EXP2_NAME)
        
        if not log1.steps or not log2.steps:
            print("Error: No valid data found in log files")
            return
        
        # 打印统计摘要
        print_summary_stats(log1, log2)
        
        # 创建对比图
        print("\nGenerating comparison plot...")
        create_comparison_plot(log1, log2, OUTPUT_PATH)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()