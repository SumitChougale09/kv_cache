import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    def __init__(self, model_name="distilgpt2", device=None):
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)

        self.model.eval()

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device
