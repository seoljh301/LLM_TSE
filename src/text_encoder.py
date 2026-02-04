import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import logging

class LLMTextEncoder(nn.Module):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", out_dim: int = 128, 
                 use_lora: bool = False, load_in_4bit: bool = False):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.out_dim = out_dim
        self.use_lora = use_lora
        self.load_in_4bit = load_in_4bit
        self.llm = None # Lazy loading
        self.tok = None
        self.proj = nn.Linear(4096, out_dim) 

    def _load_model(self):
        self.logger.info(f"Lazy loading LLM: {self.model_name}")
        # LLaMA 등 최신 모델을 위한 토크나이저 설정
        try:
            self.tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer for {self.model_name}: {e}")
            raise

        # 모델 로딩 (4bit 양자화 지원)
        quantization_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                self.load_in_4bit = False
        
        try:
            # GPU Loading: Utilize A6000 VRAM for faster processing and LoRA training
            # Force float16 to save memory (reduces usage from ~28GB to ~14GB)
            self.llm = AutoModel.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="cuda" 
            )
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}.")
            raise e

        # Paper Alignment: Frozen LLM vs LoRA (SOTA)
        if self.use_lora:
            from peft import LoraConfig, get_peft_model
            # Paper specifies: "apply the LoRA adapters to only modify keys and queries"
            lora_config = LoraConfig(
                r=16, 
                lora_alpha=32, 
                target_modules=["q_proj", "k_proj"],
                lora_dropout=0.05,
                bias="none", 
                task_type="FEATURE_EXTRACTION"
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        else:
            # Paper Alignment: Frozen LLM
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval() 

    def forward(self, texts: list[str]):
        if self.llm is None:
            self._load_model()

        device = self.llm.device # GPU
        target_device = self.proj.weight.device # GPU
        
        # Tokenizing (on GPU)
        enc = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # LLM Forward (on GPU)
        # Enable grad only if LoRA is used and we are training
        with torch.set_grad_enabled(self.use_lora and self.training):
            outputs = self.llm(**enc)
            h = outputs.last_hidden_state # (B, T, H)
        
        # Last Token Pooling (on GPU)
        sequence_lengths = enc["attention_mask"].sum(dim=1) - 1
        batch_size = h.shape[0]
        pooled = h[torch.arange(batch_size, device=device), sequence_lengths]
        
        # Move result to GPU for Projection and match dtype
        return self.proj(pooled.to(device=target_device, dtype=self.proj.weight.dtype))