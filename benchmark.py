"""
KV Cache Benchmark Suite
========================
Runs cached vs non-cached generation at multiple sequence lengths.
Includes warm-up pass and multi-run averaging for consistent results.
Outputs benchmark_results.json for chart generation.
"""

import json
import time
import torch
from model_loader import ModelLoader
from kv_cache import KVCacheManager
from generator import Generator
from profiler import get_ram_usage


NUM_RUNS = 3  # Average over this many runs per data point


def warmup(generator, cache_manager, prompt):
    print("  Warming up (compiling MPS shaders)...")
    cache_manager.reset()
    generator.generate_with_cache(prompt, max_tokens=5)
    cache_manager.reset()
    generator.generate_without_cache(prompt, max_tokens=5)
    cache_manager.reset()
    print("  Warm-up complete.\n")


def benchmark_single(generator, cache_manager, prompt, max_tokens):
    cache_manager.reset()
    _, latency_cached = generator.generate_with_cache(prompt, max_tokens)
    cache_mb = cache_manager.cache_size()
    tps_cached = max_tokens / latency_cached
    cache_manager.reset()
    _, latency_nocache = generator.generate_without_cache(prompt, max_tokens)
    tps_nocache = max_tokens / latency_nocache

    return {
        "latency_cached": latency_cached,
        "latency_nocache": latency_nocache,
        "tps_cached": tps_cached,
        "tps_nocache": tps_nocache,
        "cache_mb": cache_mb,
    }


def run_benchmark(generator, cache_manager, prompt, token_counts):
    results = []

    for max_tokens in token_counts:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {max_tokens} tokens ({NUM_RUNS} runs)")
        print(f"{'='*50}")

        run_data = []
        for run in range(NUM_RUNS):
            metrics = benchmark_single(generator, cache_manager, prompt, max_tokens)
            run_data.append(metrics)
            print(f"  Run {run+1}: cached={metrics['latency_cached']:.4f}s  "
                  f"no_cache={metrics['latency_nocache']:.4f}s")

        avg_cached = sum(r["latency_cached"] for r in run_data) / NUM_RUNS
        avg_nocache = sum(r["latency_nocache"] for r in run_data) / NUM_RUNS
        avg_tps_cached = sum(r["tps_cached"] for r in run_data) / NUM_RUNS
        avg_tps_nocache = sum(r["tps_nocache"] for r in run_data) / NUM_RUNS
        cache_mb = run_data[-1]["cache_mb"]

        speedup = avg_nocache / avg_cached if avg_cached > 0 else 0

        print(f"  ─────────────────────────────────────")
        print(f"  AVG Cached:   {avg_cached:.4f}s | {avg_tps_cached:.1f} tok/s")
        print(f"  AVG No Cache: {avg_nocache:.4f}s | {avg_tps_nocache:.1f} tok/s")
        print(f"  Speedup:      {speedup:.2f}x | Cache: {cache_mb:.3f} MB")

        results.append({
            "max_tokens": max_tokens,
            "latency_cached": round(avg_cached, 5),
            "latency_nocache": round(avg_nocache, 5),
            "tps_cached": round(avg_tps_cached, 2),
            "tps_nocache": round(avg_tps_nocache, 2),
            "speedup": round(speedup, 2),
            "cache_mb": round(cache_mb, 4),
        })

    return results


def main():
    print("=" * 60)
    print("  KV CACHE BENCHMARK SUITE")
    print(f"  Model: DistilGPT2 | Greedy Decoding | {NUM_RUNS} runs/point")
    print("=" * 60)

    loader = ModelLoader()
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    device = loader.get_device()

    cache_manager = KVCacheManager()
    generator = Generator(model, tokenizer, device, cache_manager)

    prompt = "The future of artificial intelligence depends on"

    token_counts = [10, 25, 50, 100, 150, 200]

    print(f"\nPrompt: '{prompt}'")
    print(f"Sequence lengths: {token_counts}")
    print(f"RAM at start: {get_ram_usage():.1f} MB\n")

    warmup(generator, cache_manager, prompt)

    results = run_benchmark(generator, cache_manager, prompt, token_counts)
    output = {
        "model": "distilgpt2",
        "device": device,
        "prompt": prompt,
        "num_runs": NUM_RUNS,
        "results": results,
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("Results saved to benchmark_results.json")
    print(f"{'='*60}")

    print(f"\n{'Tokens':<10}{'Cached (s)':<14}{'No Cache (s)':<14}{'Speedup':<10}{'Cache MB':<10}")
    print("-" * 58)
    for r in results:
        print(f"{r['max_tokens']:<10}{r['latency_cached']:<14}{r['latency_nocache']:<14}{r['speedup']:<10}{r['cache_mb']:<10}")


if __name__ == "__main__":
    main()
