import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def model_fn(model_dir, *args):
    # Load model from HuggingFace Hub
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
      model_dir,
      device_map="auto",
      quantization_config=bnb_config
  )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def predict_fn(data, model_and_tokenizer, *args):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    # Tokenize sentences
    sentences = data.pop("inputs", data)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
    output_sequences = model.generate(**inputs, max_new_tokens=1024)
    return tokenizer.batch_decode(output_sequences)