agent = Agent(
    model="ollama:llama3.1:8b-instruct-q8_0",
    system_prompt="""You are a helpful CSV data analysis assistant. 
    Your job is to answer questions about CSV data by analyzing it and providing 
    accurate statistics and insights. Always explain your reasoning clearly."""
)
#   File "d:\Workspaces Codes and Projects\Question-Answering-CSV-LLM Project\demo\demo_1.py", line 31, in <module>
#     agent = Agent(
#             ^^^^^^
#   File "D:\Workspaces Codes and Projects\Question-Answering-CSV-LLM Project\.venv\Lib\site-packages\pydantic_ai\agent.py", line 193, in __init__
#     self.model = models.infer_model(model)
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\Workspaces Codes and Projects\Question-Answering-CSV-LLM Project\.venv\Lib\site-packages\pydantic_ai\models\__init__.py", line 419, in infer_model
#     raise UserError(f'Unknown model: {model}')
# pydantic_ai.exceptions.UserError: Unknown model: ollama:llama3.1:8b-instruct-q8_0