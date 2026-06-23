"""Plotting utilities for experiment result visualization."""
import matplotlib.pyplot as plt
import os


def plot_loss_curves(results, title, filename, caption=None):
    """Plot training loss curves for multiple optimizers.

    Args:
        results: Dict mapping optimizer name to list of per-epoch losses.
        title: Plot title string.
        filename: Output filename (saved under results/ directory).
        caption: Optional italic caption text below the figure.
    """
    plt.rcParams.update({'font.size': 13})
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, losses in results.items():
        plt.plot(losses, label=label)
    
    plt.title(title)
    plt.xlabel('Iterations over entire dataset')
    plt.ylabel('Training cost')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    if caption is not None:
        plt.figtext(0.5, -0.02, caption, ha='center', fontsize=10, style='italic')
    plt.savefig(f"results/{filename}", dpi=150, bbox_inches='tight')
    plt.close()


def plot_vae_ablation(results_10, results_100, x_values, filename, caption=None):
    """Plot VAE bias-correction ablation as a two-panel figure.

    Args:
        results_10: Dict of {label: [losses]} after 10 epochs.
        results_100: Dict of {label: [losses]} after 100 epochs.
        x_values: X-axis values (log10 of learning rates).
        filename: Output filename (saved under results/ directory).
        caption: Optional italic caption text below the figure.
    """
    plt.rcParams.update({'font.size': 13})
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
    if caption is not None:
        fig.text(0.5, -0.05, caption, ha='center', fontsize=10, style='italic')
    plt.savefig(f"results/{filename}", dpi=150, bbox_inches='tight')
    plt.close()


def plot_cnn_two_panel(results, filename, caption=None):
    """Plot CNN training loss as a two-panel figure (paper Figure 3 style).

    Left panel shows the first 3 epochs (early convergence detail),
    right panel shows all epochs. Both use log y-scale.

    Args:
        results: Dict mapping optimizer name to list of per-epoch losses.
        filename: Output filename (saved under results/ directory).
        caption: Optional italic caption text below the figure.
    """
    plt.rcParams.update({'font.size': 13})
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for label, losses in results.items():
        # Left panel: first 3 epochs (indices 0, 1, 2)
        ax1.plot(range(1, min(4, len(losses) + 1)), losses[:3], label=label, marker='o')
        # Right panel: all epochs
        ax2.plot(range(1, len(losses) + 1), losses, label=label)
    
    ax1.set_title("CNN on CIFAR-10 (first 3 epochs)")
    ax1.set_xlabel('Iterations over entire dataset')
    ax1.set_ylabel('Training cost')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    ax2.set_title("CNN on CIFAR-10 (all epochs)")
    ax2.set_xlabel('Iterations over entire dataset')
    ax2.set_ylabel('Training cost')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    if caption is not None:
        fig.text(0.5, -0.02, caption, ha='center', fontsize=10, style='italic')
    plt.savefig(f"results/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
