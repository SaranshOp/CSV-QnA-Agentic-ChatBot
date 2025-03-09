from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Define a simple output structure
class SimpleResponse(BaseModel):
    answer: str

# Configure Ollama to work with Pydantic AI using OpenAI-compatible interface
ollama_model = OpenAIModel(
    model_name='llama3.1:8b-instruct-q8_0',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

# Create the agent
agent = Agent(ollama_model, result_type=SimpleResponse)

# Test with a simple query
result = agent.run_sync('What is Life in three sentences')
print(result.data)
