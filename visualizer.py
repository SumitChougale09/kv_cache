class CacheVisualizer:
    """
    Visualizes the token generation process, showing which tokens are 
    retrieved from cache (HIT) and which are computed (MISS).
    """
    def __init__(self):
        # ANSI colors
        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"
        self.BLUE = "\033[94m"

    def log_step(self, step_num, token, is_cache_hit, is_prefill=False):
        """
        Logs a single step of generation.
        """
        status = "HIT " if is_cache_hit else "MISS"
        color = self.GREEN if is_cache_hit else self.RED
        
        if is_prefill:
            status = "PREFILL"
            color = self.BLUE
            
        print(f"Step {step_num}: Token '{token}' -> [{color}{status}{self.RESET}]")

    def log_attention(self, full_sequence, current_token_index):
        """
        Visualizes the 'attention' window.
        """
        # Shows [HIT][HIT][HIT]...[MISS/COMPUTE]
        visual = []
        for i in range(len(full_sequence)):
            if i < current_token_index:
                visual.append(f"{self.GREEN}■{self.RESET}") # Cached
            else:
                visual.append(f"{self.RED}■{self.RESET}")   # Computed
        
        print("Attention Pattern: " + "".join(visual))
