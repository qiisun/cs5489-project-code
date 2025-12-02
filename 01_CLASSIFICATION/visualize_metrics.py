# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path
# import argparse

# def load_pt_file(file_path):
#     """Load .pt file and extract training logs"""
#     try:
#         data = torch.load(file_path, map_location=torch.device('cpu'))
        
#         # Check data format
#         if 'logs' in data:
#             logs = data['logs']
#             print(f"Successfully loaded: {file_path}")
#             print(f"Number of epochs: {len(logs)}")
#             print(f"Log fields: {logs[0].keys() if logs else 'No data'}")
#             return logs
#         else:
#             print(f"Warning: No 'logs' key found in {file_path}")
#             return None
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None

# def extract_metrics(logs):
#     """Extract metrics from logs"""
#     epochs = []
#     val_accs = []
#     val_losses = []
#     train_accs = []
#     train_losses = []
    
#     for log in logs:
#         epochs.append(log.get('epoch', len(epochs)))
#         val_accs.append(log.get('val_acc', 0.0))
#         val_losses.append(log.get('val_loss', 0.0))
#         train_accs.append(log.get('train_acc', 0.0))
#         train_losses.append(log.get('train_loss', 0.0))
    
#     return {
#         'epochs': np.array(epochs),
#         'val_accs': np.array(val_accs),
#         'val_losses': np.array(val_losses),
#         'train_accs': np.array(train_accs),
#         'train_losses': np.array(train_losses)
#     }

# def find_key_points(metrics):
#     """Find key points in training"""
#     val_accs = metrics['val_accs']
#     val_losses = metrics['val_losses']
    
#     # Find highest validation accuracy
#     max_val_acc_idx = np.argmax(val_accs)
#     max_val_acc = val_accs[max_val_acc_idx]
#     max_val_acc_epoch = metrics['epochs'][max_val_acc_idx]
    
#     # Find lowest validation loss
#     min_val_loss_idx = np.argmin(val_losses)
#     min_val_loss = val_losses[min_val_loss_idx]
#     min_val_loss_epoch = metrics['epochs'][min_val_loss_idx]
    
#     # Find convergence point (when accuracy change is below threshold)
#     convergence_threshold = 0.001
#     converged = False
#     convergence_epoch = len(val_accs) - 1
    
#     for i in range(10, len(val_accs)):
#         if i > 0 and abs(val_accs[i] - val_accs[i-1]) < convergence_threshold:
#             # Check if stable for 5 consecutive epochs
#             stable = True
#             for j in range(max(0, i-5), min(len(val_accs), i+1)):
#                 if j > 0 and abs(val_accs[j] - val_accs[j-1]) >= convergence_threshold:
#                     stable = False
#                     break
#             if stable:
#                 convergence_epoch = metrics['epochs'][i]
#                 converged = True
#                 break
    
#     return {
#         'max_val_acc': (max_val_acc_epoch, max_val_acc),
#         'min_val_loss': (min_val_loss_epoch, min_val_loss),
#         'convergence': (convergence_epoch, val_accs[np.where(metrics['epochs'] == convergence_epoch)[0][0]] if converged else None)
#     }

# def plot_metrics(metrics_list, labels, file_names):
#     """Plot metrics curves with English labels"""
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle('Training Metrics Visualization - Key Points Marked', fontsize=16, fontweight='bold')
    
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
#     markers = ['o', 's', '^', 'D']
    
#     # 1. Validation Accuracy Curve
#     ax1 = axes[0, 0]
#     for idx, (metrics, label, color, marker) in enumerate(zip(metrics_list, labels, colors, markers)):
#         epochs = metrics['epochs']
#         val_accs = metrics['val_accs']
        
#         ax1.plot(epochs, val_accs, label=label, color=color, linewidth=2, alpha=0.8)
        
#         # Mark key points
#         key_points = find_key_points(metrics)
        
#         # Highest accuracy point
#         epoch_max, acc_max = key_points['max_val_acc']
#         ax1.scatter(epoch_max, acc_max, color=color, s=100, marker=marker, 
#                    edgecolors='black', linewidth=2, zorder=5)
#         ax1.annotate(f'Max: {acc_max:.4f}\nEpoch: {epoch_max}', 
#                     xy=(epoch_max, acc_max), xytext=(10, 10),
#                     textcoords='offset points', ha='left', va='bottom',
#                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
#                     fontsize=9)
        
#         # Convergence point
#         epoch_conv, acc_conv = key_points['convergence']
#         if acc_conv is not None:
#             ax1.scatter(epoch_conv, acc_conv, color=color, s=100, marker='*', 
#                        edgecolors='black', linewidth=2, zorder=5)
#             ax1.annotate(f'Converged\nEpoch: {epoch_conv}', 
#                         xy=(epoch_conv, acc_conv), xytext=(10, -10),
#                         textcoords='offset points', ha='left', va='top',
#                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
#                         fontsize=9)
    
#     ax1.set_xlabel('Epoch', fontsize=12)
#     ax1.set_ylabel('Validation Accuracy', fontsize=12)
#     ax1.set_title('Validation Accuracy Curve', fontsize=14, fontweight='bold')
#     ax1.legend(fontsize=10)
#     ax1.grid(True, alpha=0.3)
#     ax1.set_ylim(0.5, 1.0)  # Assuming accuracy between 0.5-1.0
    
#     # 2. Validation Loss Curve
#     ax2 = axes[0, 1]
#     for idx, (metrics, label, color, marker) in enumerate(zip(metrics_list, labels, colors, markers)):
#         epochs = metrics['epochs']
#         val_losses = metrics['val_losses']
        
#         ax2.plot(epochs, val_losses, label=label, color=color, linewidth=2, alpha=0.8)
        
#         # Mark key points
#         key_points = find_key_points(metrics)
        
#         # Lowest loss point
#         epoch_min, loss_min = key_points['min_val_loss']
#         ax2.scatter(epoch_min, loss_min, color=color, s=100, marker=marker,
#                    edgecolors='black', linewidth=2, zorder=5)
#         ax2.annotate(f'Min: {loss_min:.4f}\nEpoch: {epoch_min}', 
#                     xy=(epoch_min, loss_min), xytext=(10, 10),
#                     textcoords='offset points', ha='left', va='bottom',
#                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
#                     fontsize=9)
    
#     ax2.set_xlabel('Epoch', fontsize=12)
#     ax2.set_ylabel('Validation Loss', fontsize=12)
#     ax2.set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
#     ax2.legend(fontsize=10)
#     ax2.grid(True, alpha=0.3)
    
#     # 3. Training vs Validation Accuracy Comparison
#     ax3 = axes[1, 0]
#     for idx, (metrics, label, color, _) in enumerate(zip(metrics_list, labels, colors, markers)):
#         epochs = metrics['epochs']
#         train_accs = metrics['train_accs']
#         val_accs = metrics['val_accs']
        
#         ax3.plot(epochs, train_accs, label=f'{label} (Train)', color=color, 
#                 linewidth=2, alpha=0.6, linestyle='--')
#         ax3.plot(epochs, val_accs, label=f'{label} (Val)', color=color, 
#                 linewidth=2, alpha=0.8)
    
#     ax3.set_xlabel('Epoch', fontsize=12)
#     ax3.set_ylabel('Accuracy', fontsize=12)
#     ax3.set_title('Training vs Validation Accuracy Comparison', fontsize=14, fontweight='bold')
#     ax3.legend(fontsize=10)
#     ax3.grid(True, alpha=0.3)
#     ax3.set_ylim(0.5, 1.0)
    
#     # 4. Training vs Validation Loss Comparison
#     ax4 = axes[1, 1]
#     for idx, (metrics, label, color, _) in enumerate(zip(metrics_list, labels, colors, markers)):
#         epochs = metrics['epochs']
#         train_losses = metrics['train_losses']
#         val_losses = metrics['val_losses']
        
#         ax4.plot(epochs, train_losses, label=f'{label} (Train)', color=color, 
#                 linewidth=2, alpha=0.6, linestyle='--')
#         ax4.plot(epochs, val_losses, label=f'{label} (Val)', color=color, 
#                 linewidth=2, alpha=0.8)
    
#     ax4.set_xlabel('Epoch', fontsize=12)
#     ax4.set_ylabel('Loss', fontsize=12)
#     ax4.set_title('Training vs Validation Loss Comparison', fontsize=14, fontweight='bold')
#     ax4.legend(fontsize=10)
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     # Add overall statistics
#     stats_text = ""
#     for idx, (metrics, label, file_name) in enumerate(zip(metrics_list, labels, file_names)):
#         key_points = find_key_points(metrics)
#         stats_text += f"{label} ({Path(file_name).name}):\n"
#         stats_text += f"  Highest Validation Accuracy: {key_points['max_val_acc'][1]:.4f} (Epoch {key_points['max_val_acc'][0]})\n"
#         stats_text += f"  Lowest Validation Loss: {key_points['min_val_loss'][1]:.4f} (Epoch {key_points['min_val_loss'][0]})\n"
#         if key_points['convergence'][1] is not None:
#             stats_text += f"  Convergence Point: Epoch {key_points['convergence'][0]}\n"
#         stats_text += "\n"
    
#     fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
#     return fig

# def main():
#     parser = argparse.ArgumentParser(description='Visualize training metrics from .pt files')
#     parser.add_argument('files', nargs='+', help='Paths to .pt files to visualize')
#     parser.add_argument('--labels', nargs='+', help='Labels for each file (optional)')
#     parser.add_argument('--output', type=str, default='training_metrics.png', 
#                        help='Output image filename')
    
#     args = parser.parse_args()
    
#     # Load data
#     logs_list = []
#     for file_path in args.files:
#         logs = load_pt_file(file_path)
#         if logs is not None:
#             logs_list.append(logs)
    
#     if not logs_list:
#         print("Error: No files were successfully loaded")
#         return
    
#     # Extract metrics
#     metrics_list = [extract_metrics(logs) for logs in logs_list]
    
#     # Set labels
#     if args.labels and len(args.labels) == len(args.files):
#         labels = args.labels
#     else:
#         labels = [f'Model {i+1}' for i in range(len(metrics_list))]
    
#     # Create plots
#     fig = plot_metrics(metrics_list, labels, args.files)
    
#     # Save figure
#     plt.savefig(args.output, dpi=300, bbox_inches='tight')
#     print(f"\nFigure saved to: {args.output}")
    
#     # Display figure
#     plt.show()

# if __name__ == "__main__":
#     main()


import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_pt_file(file_path):
    """Load .pt file and extract training logs"""
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'))
        
        # Check data format
        if 'logs' in data:
            logs = data['logs']
            print(f"Successfully loaded: {file_path}")
            print(f"Total number of epochs: {len(logs)}")
            print(f"Log fields: {logs[0].keys() if logs else 'No data'}")
            return logs
        else:
            print(f"Warning: No 'logs' key found in {file_path}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_last_7_epochs(logs):
    """Extract the 7 epochs from logs"""
    if len(logs) < 7:
        print(f"Warning: Only {len(logs)} epochs available, using all")
        last_epochs = logs
    else:
        last_epochs = logs[-7:]
    
    print(f"Using last {len(last_epochs)} epochs")
    
    epochs = []
    val_accs = []
    val_losses = []
    train_accs = []
    train_losses = []
    
    for i, log in enumerate(last_epochs):
        # Use relative epoch numbers (1-7)
        epochs.append(i + 1)
        val_accs.append(log.get('val_acc', 0.0))
        val_losses.append(log.get('val_loss', 0.0))
        train_accs.append(log.get('train_acc', 0.0))
        train_losses.append(log.get('train_loss', 0.0))
    
    return {
        'epochs': np.array(epochs),
        'val_accs': np.array(val_accs),
        'val_losses': np.array(val_losses),
        'train_accs': np.array(train_accs),
        'train_losses': np.array(train_losses),
        'original_epochs': [log.get('epoch', 0) for log in last_epochs]
    }

def find_key_points_last_7(metrics):
    """Find key points in the 7 epochs"""
    val_accs = metrics['val_accs']
    val_losses = metrics['val_losses']
    relative_epochs = metrics['epochs']
    original_epochs = metrics['original_epochs']
    
    # Find highest validation accuracy in 7 epochs
    max_val_acc_idx = np.argmax(val_accs)
    max_val_acc = val_accs[max_val_acc_idx]
    max_val_acc_epoch = relative_epochs[max_val_acc_idx]
    max_val_acc_original_epoch = original_epochs[max_val_acc_idx]
    
    # Find lowest validation loss in 7 epochs
    min_val_loss_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_loss_idx]
    min_val_loss_epoch = relative_epochs[min_val_loss_idx]
    min_val_loss_original_epoch = original_epochs[min_val_loss_idx]
    
    # Check if it's the final epoch
    is_final_epoch = len(val_accs) - 1
    
    return {
        'max_val_acc': (max_val_acc_epoch, max_val_acc, max_val_acc_original_epoch),
        'min_val_loss': (min_val_loss_epoch, min_val_loss, min_val_loss_original_epoch),
        'final_epoch': (relative_epochs[-1], val_accs[-1], original_epochs[-1])
    }

def plot_last_7_epochs(metrics_list, labels, file_names):
    """Plot metrics for the 7 epochs with English labels"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('7 Epochs Training Metrics Visualization', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '--', '-.', ':']
    
    # 1. Validation Accuracy Curve (7 epochs)
    ax1 = axes[0, 0]
    for idx, (metrics, label, color, marker, line_style) in enumerate(zip(metrics_list, labels, colors, markers, line_styles)):
        epochs = metrics['epochs']
        val_accs = metrics['val_accs']
        original_epochs = metrics['original_epochs']
        
        ax1.plot(epochs, val_accs, label=label, color=color, linewidth=2.5, 
                alpha=0.9, linestyle=line_style, marker=marker, markersize=8)
        
        # Mark key points
        key_points = find_key_points_last_7(metrics)
        
        # Highest accuracy point
        epoch_max, acc_max, orig_epoch_max = key_points['max_val_acc']
        ax1.scatter(epoch_max, acc_max, color=color, s=120, marker='*', 
                   edgecolors='black', linewidth=2, zorder=10)
        ax1.annotate(f'Highest: {acc_max:.4f}\nEpoch: {orig_epoch_max}', 
                    xy=(epoch_max, acc_max), xytext=(15, 15),
                    textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    fontsize=10, fontweight='bold')
        
        # Final epoch point
        epoch_final, acc_final, orig_epoch_final = key_points['final_epoch']
        ax1.scatter(epoch_final, acc_final, color=color, s=120, marker='D', 
                   edgecolors='black', linewidth=2, zorder=10)
        ax1.annotate(f'Final: {acc_final:.4f}', 
                    xy=(epoch_final, acc_final), xytext=(15, -15),
                    textcoords='offset points', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Relative Epoch (7 Epochs)', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Validation Accuracy - 7 Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show 1-7
    if metrics_list:
        max_epochs = max([len(m['epochs']) for m in metrics_list])
        ax1.set_xlim(0.5, max_epochs + 0.5)
        ax1.set_xticks(range(1, max_epochs + 1))
    
    # 2. Validation Loss Curve (7 epochs)
    ax2 = axes[0, 1]
    for idx, (metrics, label, color, marker, line_style) in enumerate(zip(metrics_list, labels, colors, markers, line_styles)):
        epochs = metrics['epochs']
        val_losses = metrics['val_losses']
        original_epochs = metrics['original_epochs']
        
        ax2.plot(epochs, val_losses, label=label, color=color, linewidth=2.5, 
                alpha=0.9, linestyle=line_style, marker=marker, markersize=8)
        
        # Mark key points
        key_points = find_key_points_last_7(metrics)
        
        # Lowest loss point
        epoch_min, loss_min, orig_epoch_min = key_points['min_val_loss']
        ax2.scatter(epoch_min, loss_min, color=color, s=120, marker='*',
                   edgecolors='black', linewidth=2, zorder=10)
        ax2.annotate(f'Lowest: {loss_min:.4f}\nEpoch: {orig_epoch_min}', 
                    xy=(epoch_min, loss_min), xytext=(15, 15),
                    textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    fontsize=10, fontweight='bold')
        
        # Final epoch point
        epoch_final, _, orig_epoch_final = key_points['final_epoch']
        loss_final = val_losses[-1]
        ax2.scatter(epoch_final, loss_final, color=color, s=120, marker='D',
                   edgecolors='black', linewidth=2, zorder=10)
        ax2.annotate(f'Final: {loss_final:.4f}', 
                    xy=(epoch_final, loss_final), xytext=(15, -15),
                    textcoords='offset points', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Relative Epoch (7 Epochs)', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss - 7 Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    if metrics_list:
        max_epochs = max([len(m['epochs']) for m in metrics_list])
        ax2.set_xlim(0.5, max_epochs + 0.5)
        ax2.set_xticks(range(1, max_epochs + 1))
    
    # 3. Training vs Validation Accuracy (7 Epochs)
    ax3 = axes[1, 0]
    for idx, (metrics, label, color, _, line_style) in enumerate(zip(metrics_list, labels, colors, markers, line_styles)):
        epochs = metrics['epochs']
        train_accs = metrics['train_accs']
        val_accs = metrics['val_accs']
        
        ax3.plot(epochs, train_accs, label=f'{label} (Train)', color=color, 
                linewidth=2, alpha=0.7, linestyle='--', marker='o', markersize=6)
        ax3.plot(epochs, val_accs, label=f'{label} (Val)', color=color, 
                linewidth=2.5, alpha=0.9, linestyle='-', marker='s', markersize=6)
    
    ax3.set_xlabel('Relative Epoch (7 Epochs)', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Training vs Validation Accuracy - 7 Epochs', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='best', ncol=2)
    ax3.grid(True, alpha=0.3)
    
    if metrics_list:
        max_epochs = max([len(m['epochs']) for m in metrics_list])
        ax3.set_xlim(0.5, max_epochs + 0.5)
        ax3.set_xticks(range(1, max_epochs + 1))
    
    # 4. Training vs Validation Loss (7 Epochs)
    ax4 = axes[1, 1]
    for idx, (metrics, label, color, _, line_style) in enumerate(zip(metrics_list, labels, colors, markers, line_styles)):
        epochs = metrics['epochs']
        train_losses = metrics['train_losses']
        val_losses = metrics['val_losses']
        
        ax4.plot(epochs, train_losses, label=f'{label} (Train)', color=color, 
                linewidth=2, alpha=0.7, linestyle='--', marker='o', markersize=6)
        ax4.plot(epochs, val_losses, label=f'{label} (Val)', color=color, 
                linewidth=2.5, alpha=0.9, linestyle='-', marker='s', markersize=6)
    
    ax4.set_xlabel('Relative Epoch (7 Epochs)', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Training vs Validation Loss - 7 Epochs', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9, loc='best', ncol=2)
    ax4.grid(True, alpha=0.3)
    
    if metrics_list:
        max_epochs = max([len(m['epochs']) for m in metrics_list])
        ax4.set_xlim(0.5, max_epochs + 0.5)
        ax4.set_xticks(range(1, max_epochs + 1))
    
    plt.tight_layout()
    
    # Add overall statistics for 7 epochs
    stats_text = "7 epochs Statistics:\n"
    for idx, (metrics, label, file_name) in enumerate(zip(metrics_list, labels, file_names)):
        key_points = find_key_points_last_7(metrics)
        original_epochs = metrics['original_epochs']
        
        stats_text += f"\n{label} ({Path(file_name).name}):\n"
        stats_text += f"  Epochs analyzed: {original_epochs[0]} to {original_epochs[-1]}\n"
        stats_text += f"  Highest Validation Accuracy: {key_points['max_val_acc'][1]:.4f} "
        stats_text += f"(Epoch {key_points['max_val_acc'][2]})\n"
        stats_text += f"  Lowest Validation Loss: {key_points['min_val_loss'][1]:.4f} "
        stats_text += f"(Epoch {key_points['min_val_loss'][2]})\n"
        stats_text += f"  Final Validation Accuracy: {key_points['final_epoch'][1]:.4f} "
        stats_text += f"(Epoch {key_points['final_epoch'][2]})\n"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize 7 epochs training metrics from .pt files')
    parser.add_argument('files', nargs='+', help='Paths to .pt files to visualize')
    parser.add_argument('--labels', nargs='+', help='Labels for each file (optional)')
    parser.add_argument('--output', type=str, default='last_7_epochs_metrics.png', 
                       help='Output image filename')
    
    args = parser.parse_args()
    
    # Load data
    logs_list = []
    for file_path in args.files:
        logs = load_pt_file(file_path)
        if logs is not None:
            logs_list.append(logs)
    
    if not logs_list:
        print("Error: No files were successfully loaded")
        return
    
    # Extract 7 epochs metrics
    metrics_list = [extract_last_7_epochs(logs) for logs in logs_list]
    
    # Set labels
    if args.labels and len(args.labels) == len(args.files):
        labels = args.labels
    else:
        labels = [f'Model {i+1}' for i in range(len(metrics_list))]
    
    # Create plots
    fig = plot_last_7_epochs(metrics_list, labels, args.files)
    
    # Save figure
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {args.output}")
    
    # Display figure
    plt.show()

if __name__ == "__main__":
    main()