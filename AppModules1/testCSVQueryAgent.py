import pandas as pd
from llm_agent import CSVQueryAgent

# Create a sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
summary = {'rows': 3, 'columns': 2, 'column_names': ['A', 'B']}

# Initialize the agent
agent = CSVQueryAgent()

# Test a query
result, error = agent.process_query(df, summary, "What is the sum of column A?")
print(result.answer if not error else error)
