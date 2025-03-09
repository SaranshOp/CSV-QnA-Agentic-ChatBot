import ollama

# Simple test to see if Ollama is working
response = ollama.chat(
    model='llama3.1:8b-instruct-q8_0',
    messages=[
        {
            'role': 'user',
            'content': 'What is life'
        }
    ]
)

print(response['message']['content'])
