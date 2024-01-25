import ollama

foo = ollama.embeddings(model='llama2', prompt='They sky is blue because of rayleigh scattering')
print(len(foo))
print(type(foo))
print(list(foo.keys()))
print(len(foo['embedding'])) # 4096
print(type(foo['embedding'])) # list
print(type(foo['embedding'][0])) # float
