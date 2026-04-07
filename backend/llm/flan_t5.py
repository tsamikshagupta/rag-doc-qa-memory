"""
Local text generation using FLAN-T5.
"""
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from backend.utils.config import GENERATION_MODEL

logger = logging.getLogger(__name__)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading FLAN-T5 '{GENERATION_MODEL}' on {device_str}...")

tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    GENERATION_MODEL, 
    torch_dtype=torch.float16 if device_str == "cuda" else torch.float32
).to(device_str)

def generate_text(prompt: str, max_length: int = 250) -> str:
    """
    Generate text using local huggingface FLAN-T5.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device_str)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
