import torch
import time


class Generator:

    def __init__(self, model, tokenizer, device, cache_manager=None, visualizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache_manager = cache_manager
        self.visualizer = visualizer

    @torch.no_grad()
    def generate_with_cache(self, prompt, max_tokens=50):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = inputs["input_ids"]

        start = time.time()

        # PREFILL: First pass processes the FULL prompt to build the KV cache
        outputs = self.model(
            input_ids=generated,
            past_key_values=None,
            use_cache=True
        )
        # Log Prefill
        if self.visualizer:
            print("\n--- PREFILL PHASE ---")
            for i, token_id in enumerate(generated[0]):
                token_str = self.tokenizer.decode([token_id])
                self.visualizer.log_step(0, token_str, is_cache_hit=False, is_prefill=True)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        self.cache_manager.update(outputs)

        # DECODE: Subsequent passes feed only the new token + cached KV
        if self.visualizer:
            print("\n--- DECODE PHASE ---")

        for step in range(max_tokens - 1):

            outputs = self.model(
                input_ids=next_token,
                past_key_values=self.cache_manager.get(),
                use_cache=True
            )

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            self.cache_manager.update(outputs)

            if self.visualizer:
               token_str = self.tokenizer.decode(next_token[0]) 
               # We hit the cache for all past tokens, computed the new one
               # Visualization: show full sequence state
               self.visualizer.log_attention(generated[0], len(generated[0]) - 1)
               print(f" Generated: '{token_str}'")

        latency = time.time() - start

        text = self.tokenizer.decode(generated[0])

        return text, latency

    @torch.no_grad()
    def generate_without_cache(self, prompt, max_tokens=50):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = inputs["input_ids"]

        start = time.time()

        for _ in range(max_tokens):

            outputs = self.model(
                input_ids=generated,
                use_cache=False
            )

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

        latency = time.time() - start

        text = self.tokenizer.decode(generated[0])

        return text, latency
