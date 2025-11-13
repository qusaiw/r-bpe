from vllm import LLM, SamplingParams 
from vllm.transformers_utils.tokenizer_base import TokenizerRegistry 
import rbpe


# Initialize the LLM
llm = LLM(
    model="/workspace/cr7b-tinybox-exp", 
    trust_remote_code=True
)  # tokenizer_mode="custom",

# Set up sampling parameters
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.9, 
    max_tokens=256
)

# Get the tokenizer from the LLM
tokenizer = llm.get_tokenizer()

print("Chat started. Type 'quit' or 'exit' to end the conversation.\n")

while True:
    prompt = input("User: ")
    
    if prompt.lower() in ['quit', 'exit']:
        print("Ending chat...")
        break
    
    # Format as chat messages
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    # Print the response
    response = outputs[0].outputs[0].text
    print(f"Assistant: {response}\n")
