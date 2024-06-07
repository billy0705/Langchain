import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import time


def get_qmodel_tokenizer(model_name, hf_token):

    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and True:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16")
            print("=" * 80)

    device_map = {"": 0}

    # Load LLaMA base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=hf_token
    )
    model.config.use_cache = False

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=hf_token
    )

    return model, tokenizer


if __name__ == "__main__":
    hf_token = ""
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    model, tokenizer = get_qmodel_tokenizer(model_name, hf_token)
    prompt = "Hello, give me 3 cities name in Taiwan. And one site in the city"
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200
    )
    start_time = time.time()
    result = pipe(f"You are a chatbot who always responds in english!{prompt}")
    end_time = time.time()
    print("Time: ", end_time - start_time)
    print(result[0]['generated_text'])
