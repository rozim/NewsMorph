from llama_cpp import Llama
llm = Llama(
  model_path='/Users/dave/Projects/Models/mistral-7b-v0.1.Q5_K_M.gguf',
  use_mlock=True,
  # n_gpu_layers=-1, # Uncomment to use GPU acceleration
  # seed=1337, # Uncomment to set a specific seed
  # n_ctx=2048, # Uncomment to increase the context window
)
output = llm(
  'Continue the news article with the title "What It Took Young People in China to Get Their Jobs"',
  max_tokens=400, # Generate up to 32 tokens, set to None to generate up to the end of the context window
  #stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
  echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion

print(output)
print('\n')
print(output['choices'])
print('\n')
print(output['choices'][0])
print('\n')
print(output['choices'][0]['text'])
