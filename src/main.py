import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
def typeg(text, min_delay=0.03, max_delay=0.15):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(random.uniform(min_delay, max_delay))
    print()
def talk(model, tokenizer, device="cpu"):
    chat_history_ids = None
    print("stupid ai bot that is dumb")
    print("type 'exit' to exit hreh")
    while True:
        user_input = input("you: ")
        if user_input.lower() in ["exit", "quit"]:
            print("bai greg")
            break
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
        try:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            print("Bot: ", end='')
            typeg(reply)
        except Exception as e:
            print(f"an error occurred... {e}")
            print("plz try again later or smth")
if __name__ == "__main__":
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"using: {device}")
    talk(model, tokenizer, device)
