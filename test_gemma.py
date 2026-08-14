from llama_cpp import Llama
import sys

# Boot the model with our optimizations
llm = Llama(
    model_path="gemma-4-E2B-it-Q3_K_M.gguf",
    n_ctx=4096,
    n_threads=4,
    verbose=False  # Hides the messy loading logs
)

print("\n--- AgriGemma Q3 Interactive Terminal ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Farmer: ")
    if user_input.lower() in ['exit', 'quit']:
        break
        
    print("Gemma : ", end="", flush=True)
    
    # Stream the response back exactly like ChatGPT
    stream = llm.create_chat_completion(
        messages=[{"role": "user", "content": user_input}],
        stream=True
    )
    
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            print(delta['content'], end='', flush=True)
    print("\n")
