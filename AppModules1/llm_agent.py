import os
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryResult(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    explanation: Optional[str] = Field(None, description="Explanation of how the answer was derived")
    plot_type: Optional[str] = Field(None, description="Type of plot to generate (if applicable)")
    x_column: Optional[str] = Field(None, description="Column to use for x-axis (if plotting)")
    y_column: Optional[str] = Field(None, description="Column to use for y-axis (if plotting)")
    title: Optional[str] = Field(None, description="Title for the plot (if plotting)")

class CSVData(BaseModel):
    data: Dict[str, Any] = Field(description="Dictionary representation of the DataFrame")
    summary: Dict[str, Any] = Field(description="Summary information about the DataFrame")

class CSVQueryAgent:
    def __init__(self, model_name="llama3.1:8b-instruct-q8_0"):
        self.model_name = model_name
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        @Tool
        def get_dataframe_info(data: Dict[str, Any]) -> Dict[str, Any]:
            df = pd.DataFrame(data)
            return {
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
                "description": f"Dataset with {len(df)} rows and {len(df.columns)} columns"
            }

        @Tool
        def run_query(data: Dict[str, Any], query: str) -> Dict[str, Any]:
            df = pd.DataFrame(data)
            try:
                result = df.query(query)
                return {"result": result.head(10).to_dict(orient="records"), "count": len(result)}
            except Exception as e:
                return {"error": str(e)}

        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Create OpenAI model with Ollama provider
        ollama_model = OpenAIModel(
            model_name=self.model_name, 
            provider=OpenAIProvider(base_url=f"{ollama_base_url}/v1")
        )
        
        # Create agent with the model
        agent = Agent(
            model=ollama_model,
            result_type=QueryResult,
            deps_type=CSVData,
            tools=[get_dataframe_info, run_query],
            system_prompt="""
            You are a CSV data analysis assistant. Answer questions about CSV data accurately and suggest visualizations when appropriate.
            """
        )
        return agent

    def process_query(self, df: pd.DataFrame, summary: Dict[str, Any], query: str) -> Tuple[QueryResult, Optional[str]]:
        try:
            df_dict = df.to_dict(orient="list")
            csv_data = CSVData(data=df_dict, summary=summary)
            result = self.agent.run_sync(query, deps=csv_data)
            return result, None
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            fallback_result = QueryResult(
                answer=f"Error processing query: {str(e)}",
                explanation="There was a technical issue. Please try rephrasing your question."
            )
            return fallback_result, str(e)