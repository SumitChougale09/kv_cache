from model_loader import ModelLoader
from kv_cache import KVCacheManager
from generator import Generator
from profiler import get_ram_usage


loader = ModelLoader()

model = loader.get_model()
tokenizer = loader.get_tokenizer()
device = loader.get_device()

cache_manager = KVCacheManager()

generator = Generator(model, tokenizer, device, cache_manager)

prompt = "KV caching is important because"

print("RAM before:", get_ram_usage(), "MB")

text, latency = generator.generate_with_cache(prompt)
print("\n--- WITH CACHE ---")
print("Generated:", text)
print("Latency:", round(latency, 4), "s")
print("Cache size:", round(cache_manager.cache_size(), 4), "MB")

cache_manager.reset()

text2, latency2 = generator.generate_without_cache(prompt)
print("\n--- WITHOUT CACHE ---")
print("Generated:", text2)
print("Latency:", round(latency2, 4), "s")

print("\n--- COMPARISON ---")
print(f"Speedup: {latency2 / latency:.2f}x")
print("RAM after:", get_ram_usage(), "MB")
