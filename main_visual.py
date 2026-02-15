from model_loader import ModelLoader
from kv_cache import KVCacheManager
from generator import Generator
from visualizer import CacheVisualizer

print("Initializing Cache Visualizer Demo...")

loader = ModelLoader()
model = loader.get_model()
tokenizer = loader.get_tokenizer()
device = loader.get_device()

cache_manager = KVCacheManager()
visualizer = CacheVisualizer()

# Inject visualizer into Generator
generator = Generator(model, tokenizer, device, cache_manager, visualizer=visualizer)

prompt = "The quick brown fox jumps over the lazy"

print(f"\nPrompt: '{prompt}'")
print("Starting Generation...\n")

text, latency = generator.generate_with_cache(prompt, max_tokens=10)

print("\n\nFinal Text:", text)
