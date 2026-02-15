import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


COLORS = {
    "cached": "#00C853",      # Vibrant green
    "nocache": "#FF1744",     # Vibrant red
    "memory": "#2979FF",      # Blue
    "bg": "#0D1117",          # Dark background
    "text": "#E6EDF3",        # Light text
    "grid": "#21262D",        # Subtle grid
    "accent": "#F0883E",      # Orange accent
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["bg"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 12,
})


def load_results(path="benchmark_results.json"):
    with open(path) as f:
        return json.load(f)


def plot_latency_comparison(results, output_dir):
    data = results["results"]
    tokens = [r["max_tokens"] for r in data]
    cached = [r["latency_cached"] for r in data]
    nocache = [r["latency_nocache"] for r in data]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(tokens))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], cached, width,
                   label="With KV Cache", color=COLORS["cached"], alpha=0.9,
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar([i + width/2 for i in x], nocache, width,
                   label="Without KV Cache", color=COLORS["nocache"], alpha=0.9,
                   edgecolor="white", linewidth=0.5)

    # Add speedup labels on top
    for i, r in enumerate(data):
        ax.annotate(f'{r["speedup"]}x',
                    xy=(i + width/2, nocache[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold",
                    color=COLORS["accent"])

    ax.set_xlabel("Sequence Length (tokens)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latency (seconds)", fontsize=13, fontweight="bold")
    ax.set_title("KV Cache: Latency Comparison\n(DistilGPT2, Greedy Decoding)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.legend(fontsize=11, loc="upper left",
              facecolor=COLORS["bg"], edgecolor=COLORS["grid"])
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_scaling_curve(results, output_dir):
    data = results["results"]
    tokens = [r["max_tokens"] for r in data]
    cached = [r["latency_cached"] for r in data]
    nocache = [r["latency_nocache"] for r in data]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(tokens, cached, "o-", color=COLORS["cached"], linewidth=2.5,
            markersize=8, label="With KV Cache (≈ Linear)", zorder=5)
    ax.plot(tokens, nocache, "s-", color=COLORS["nocache"], linewidth=2.5,
            markersize=8, label="Without KV Cache (≈ Quadratic)", zorder=5)

    # Fill between to emphasize the gap
    ax.fill_between(tokens, cached, nocache,
                    alpha=0.08, color=COLORS["nocache"])

    ax.set_xlabel("Sequence Length (tokens)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latency (seconds)", fontsize=13, fontweight="bold")
    ax.set_title("KV Cache: Scaling Behavior\nWhy Production Systems Cache KV Pairs",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=11, facecolor=COLORS["bg"], edgecolor=COLORS["grid"])
    ax.grid(True, alpha=0.2)

    # Annotate the gap at the largest sequence length
    last = data[-1]
    mid_y = (last["latency_cached"] + last["latency_nocache"]) / 2
    ax.annotate(f'Gap: {last["latency_nocache"] - last["latency_cached"]:.2f}s',
                xy=(last["max_tokens"], mid_y),
                xytext=(-80, 0), textcoords="offset points",
                fontsize=11, fontweight="bold", color=COLORS["accent"],
                arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.5))

    plt.tight_layout()
    path = os.path.join(output_dir, "scaling_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_memory_growth(results, output_dir):
    data = results["results"]
    tokens = [r["max_tokens"] for r in data]
    cache_mb = [r["cache_mb"] for r in data]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(tokens, cache_mb, alpha=0.3, color=COLORS["memory"])
    ax.plot(tokens, cache_mb, "o-", color=COLORS["memory"], linewidth=2.5,
            markersize=8, label="KV Cache Size", zorder=5)

    # Annotate each point
    for t, mb in zip(tokens, cache_mb):
        ax.annotate(f'{mb:.3f}',
                    xy=(t, mb), xytext=(0, 12),
                    textcoords="offset points", ha="center",
                    fontsize=10, color=COLORS["text"])

    ax.set_xlabel("Sequence Length (tokens)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cache Memory (MB)", fontsize=13, fontweight="bold")
    ax.set_title("KV Cache: Memory Growth\nThe Tradeoff That Led to PagedAttention",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=11, facecolor=COLORS["bg"], edgecolor=COLORS["grid"])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "memory_growth.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 50)
    print("  GENERATING CHARTS")
    print("=" * 50)

    output_dir = "charts"
    os.makedirs(output_dir, exist_ok=True)

    results = load_results()
    print(f"\n  Model: {results['model']}")
    print(f"  Device: {results['device']}")
    print(f"  Data points: {len(results['results'])}\n")

    plot_latency_comparison(results, output_dir)
    plot_scaling_curve(results, output_dir)
    plot_memory_growth(results, output_dir)

    print(f"\n  All charts saved to {output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
