import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
def typeg(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(random.uniform(0.03, 0.15))
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chat_history_ids = None
def chat(prompt):
    global chat_history_ids
    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply
while True:
    user_input = input("You: ")
    reply = chat(user_input)
    print("Bot: ", end='')
    typeg(reply + "\n")
