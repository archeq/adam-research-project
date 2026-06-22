import matplotlib.pyplot as plt
import os

def plot_loss_curves(results, title, filename):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, losses in results.items():
        plt.plot(losses, label=label)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Training cost')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

def plot_vae_ablation(results_10, results_100, x_values, filename):
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for label, losses in results_10.items():
        linestyle = '-' if 'corrected' in label else '--'
        ax1.plot(x_values, losses, label=label, linestyle=linestyle, marker='o')
    ax1.set_title("(a) after 10 epochs")
    ax1.set_xlabel('log10(alpha)')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    for label, losses in results_100.items():
        linestyle = '-' if 'corrected' in label else '--'
        ax2.plot(x_values, losses, label=label, linestyle=linestyle, marker='o')
    ax2.set_title("(b) after 100 epochs")
    ax2.set_xlabel('log10(alpha)')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    # Put legend outside
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"results/{filename}")
    plt.close()
