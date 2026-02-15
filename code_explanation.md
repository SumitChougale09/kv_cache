# KV Cache Project - Deep Dive Code Explanation

This document breaks down every function in our custom KV cache implementation line-by-line. 

## 1. `model_loader.py`

Responsible for cleanly loading the model and tokenizer on the correct device.

### `ModelLoader.__init__`
```python
class ModelLoader:
    def __init__(self, model_name="distilgpt2", device=None):
        # 1. Device Selection Logic
        # If user provides a device, use it. Otherwise, auto-detect.
        # "mps" = Metal Performance Shaders (Mac M1/M2/M3 GPU).
        # "cpu" = Fail-safe default.
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        # 2. Load Tokenizer
        # Converts text -> numbers (input_ids).
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 3. Load Model
        # AutoModelForCausalLM = Standard class for GPT-style models.
        # torch_dtype=torch.float32 = We force full precision for consistency 
        # (fp16 is faster but can be tricky on some CPUs/MPS).
        # .to(self.device) = Moves the heavy model weights to GPU/CPU memory.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)

        # 4. Set to Eval Mode
        # Critical! Disables Dropout and other training-specific layers.
        # If you forget this, generation results will be random/garbage.
        self.model.eval()
```

---

## 2. `kv_cache.py`

The core logic for managing the Key-Value (KV) cache memory.

### `KVCacheManager.update`
```python
    def update(self, outputs):
        """
        Extract KV from model outputs.
        """
        # "outputs" is the object returned by model().
        # outputs.past_key_values contains the K and V matrices for every layer.
        # We store this reference to feed it back in the next step.
        self.past_key_values = outputs.past_key_values
        
        # Simple counter to track how many tokens we've processed.
        self.total_tokens += 1
```

### `KVCacheManager.cache_size`
```python
    def cache_size(self):
        """
        Rough estimator of cache memory.
        """
        # If no cache exists yet, size is 0.
        if self.past_key_values is None:
            return 0

        total_elements = 0

        # Iterate over every layer in the transformer.
        for layer in self.past_key_values:
            # Each layer is a tuple containing at least (Key, Value).
            # We use [:2] to ensure we only grab k and v, ignoring any extra metadata.
            # (This was the fix for the ValueError!)
            k, v = layer[:2]
            
            # .numel() = Number of Elements (e.g., a 2x2 matrix has 4 elements).
            total_elements += k.numel() + v.numel()

        # Calculation:
        # float32 uses 4 bytes per number.
        # 1024 ** 2 converts Bytes -> Megabytes (MB).
        return total_elements * 4 / (1024 ** 2)  # MB
```

---

## 3. `generator.py`

The engine that runs the inference loop. This contains the most critical logic difference: **Prefill vs. Decode**.

### `Generator.generate_with_cache` (Optimized)

```python
    @torch.no_grad() # Disables gradient calculation (saves massive memory)
    def generate_with_cache(self, prompt, max_tokens=50):

        # 1. Tokenize Input
        # return_tensors="pt" gives us PyTorch tensors immediately.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # This is our "growing" list of tokens (Prompt + Generated).
        generated = inputs["input_ids"]

        start = time.time()

        # === PHASE 1: PREFILL ===
        # We model the ENTIRE prompt at once.
        # inputs["input_ids"] = full prompt.
        # past_key_values=None = We have no history yet.
        outputs = self.model(
            input_ids=generated,
            past_key_values=None,
            use_cache=True # Tells model to return past_key_values
        )
        
        # Greedy Sampling: Pick the single most likely next token.
        # .logits[:, -1, :] = Look at predictions for the LAST position only.
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        
        # Append new token to our sequence.
        generated = torch.cat([generated, next_token], dim=-1)
        
        # SAVE THE CACHE! This is the magic step.
        self.cache_manager.update(outputs)

        # === PHASE 2: DECODE LOOP ===
        # Now we generate the rest, one by one.
        for _ in range(max_tokens - 1):

            # Critical Difference:
            # We ONLY feed "next_token" (1 single token), not the whole history.
            # The history is provided via `past_key_values`.
            outputs = self.model(
                input_ids=next_token, 
                past_key_values=self.cache_manager.get(), # Inject cached memory
                use_cache=True
            )

            # Same sampling logic as before
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Update cache with the NEW token's keys/values.
            self.cache_manager.update(outputs)

        latency = time.time() - start

        # Decode: Convert list of numbers back to string (e.g., [15496] -> "hello").
        text = self.tokenizer.decode(generated[0])

        return text, latency
```

### `Generator.generate_without_cache` (Slow / Naive)

```python
    @torch.no_grad()
    def generate_without_cache(self, prompt, max_tokens=50):
        # ...setup is same...

        for _ in range(max_tokens):
            # THE BOTTLENECK:
            # We feed "generated" which grows larger every step (10, 11, 12... tokens).
            # The model re-computes attention for ALL previous tokens every single time.
            # It's O(N^2) complexity vs O(N) for cached.
            outputs = self.model(
                input_ids=generated,
                use_cache=False # We ask model NOT to return cache
            )

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

        # ...
```

---

## 4. `profiler.py`

### `get_ram_usage`
```python
def get_ram_usage():
    # Helper to get the current process ID (PID)
    process = psutil.Process(os.getpid())
    
    # .rss = Resident Set Size (Physical Memory used by process)
    # / (1024 ** 2) = Convert Bytes to MB.
    return process.memory_info().rss / (1024 ** 2)
```
