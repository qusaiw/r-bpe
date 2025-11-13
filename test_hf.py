# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "/workspace/cr7b-tinybox-exp"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# input_text = "من أنت؟"
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids, max_new_tokens=32)
# print(tokenizer.decode(outputs[0]))


while True:
    prompt = input("User: ")
    messages = [
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
    
    outputs = model.generate(**input_ids, max_new_tokens=256)
    print(tokenizer.decode(outputs[0]))

