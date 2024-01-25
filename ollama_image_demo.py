import ollama

with open('mysterious-rock-orig-smaller.png', 'rb') as file:
  response = ollama.chat(
    model='llava',
    messages=[
      {
        'role': 'user',
        'content': 'Describe this image',
        'images': [file.read()],
      },
    ],
  )
print(response['message']['content'])
