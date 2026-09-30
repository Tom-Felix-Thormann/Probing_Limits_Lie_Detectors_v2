import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_BASE_DIR = SCRIPT_DIR / "model_weights"

SYSTEM_PROMPT = "You are a pirate. Answer in a pirate style."
USER_PROMPT = "Hello, how are you?"

def find_model_dir(base_dir):
    """
    Find the model directory dynamically, depending on whether the script is executed locally or on the cluster
    """

    # model_dir = [
    #     p for p in base_dir.iterdir()
    #     if p.is_dir() and "llama" in p.name.lower() and (p / "config.json").exists()
    # ][0]
    model_dir = base_dir / "llama-3.1-8B-instruct" # for test with 8b model first

    return model_dir

def main():
    model_path = find_model_dir(MODEL_BASE_DIR)
    print(f"Using model: {model_path.name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",  # spreads the model over all visible GPUs (needs `accelerate`)
        local_files_only=True,  # compute nodes have no internet
    )
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    dialogue = (
        f"Model: {model_path.name}\n\n"
        f"System: {SYSTEM_PROMPT}\n"
        f"User: {USER_PROMPT}\n"
        f"Assistant: {answer}\n"
    )
    output_file = SCRIPT_DIR / f"llm_test{model_path.name}.txt" 
    output_file.write_text(dialogue, encoding="utf-8")

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()