class KVCacheManager:
    """
    Manages past_key_values manually.
    Think of this as your mini vLLM brain.
    """

    def __init__(self):
        self.past_key_values = None
        self.total_tokens = 0

    def update(self, outputs):
        """
        Extract KV from model outputs.
        """
        self.past_key_values = outputs.past_key_values
        self.total_tokens += 1

    def get(self):
        return self.past_key_values

    def reset(self):
        self.past_key_values = None
        self.total_tokens = 0

    def cache_size(self):
        """
        Rough estimator of cache memory.
        """
        if self.past_key_values is None:
            return 0

        total_elements = 0

        for layer in self.past_key_values:
            k, v = layer[:2]
            total_elements += k.numel() + v.numel()

        # float32 = 4 bytes
        return total_elements * 4 / (1024 ** 2)  # MB
