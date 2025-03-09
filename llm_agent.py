import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent, Tool
import ollama
from utils import safe_execute, format_error_message

logger = logging.getLogger(__name__)

# Define Pydantic models for structured outputs
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    description: str = Field(description="Brief description of what this column represents")

class DataframeInfo(BaseModel):
    num_rows: int
    num_columns: int
    columns: List[ColumnInfo]
    description: str = Field(description="Brief description of the dataset")

class QueryResult(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    explanation: Optional[str] = Field(None, description="Explanation of how the answer was derived")
    plot_type: Optional[str] = Field(None, description="Type of plot to generate (if applicable)")
    x_column: Optional[str] = Field(None, description="Column to use for x-axis (if plotting)")
    y_column: Optional[str] = Field(None, description="Column to use for y-axis (if plotting)")
    title: Optional[str] = Field(None, description="Title for the plot (if plotting)")

# Use a dictionary instead of DataFrame directly
class CSVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    data: Dict[str, Any] = Field(description="Dictionary representation of the DataFrame")
    summary: Dict[str, Any] = Field(description="Summary information about the DataFrame")

class CSVQueryAgent:
    def __init__(self, model_name="llama3.1:8b-instruct-q8_0"):
        self.model_name = model_name
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create and configure the Pydantic AI agent."""
        # Define tools for the agent
        @Tool
        def get_dataframe_info(data: Dict[str, Any]) -> DataframeInfo:
            """Get information about the dataframe."""
            # Convert dict back to DataFrame for processing
            df = pd.DataFrame(data)
            
            columns = []
            for col in df.columns:
                columns.append(ColumnInfo(
                    name=col,
                    dtype=str(df[col].dtype),
                    description=f"Column with {df[col].nunique()} unique values"
                ))
            
            return DataframeInfo(
                num_rows=len(df),
                num_columns=len(df.columns),
                columns=columns,
                description=f"Dataset with {len(df)} rows and {len(df.columns)} columns"
            )
        
        @Tool
        def get_column_stats(data: Dict[str, Any], column_name: str) -> Dict[str, Any]:
            """Get statistics for a specific column."""
            # Convert dict back to DataFrame for processing
            df = pd.DataFrame(data)
            
            if column_name not in df.columns:
                return {"error": f"Column '{column_name}' not found"}
            
            col_data = df[column_name]
            result = {
                "name": column_name,
                "dtype": str(col_data.dtype),
                "unique_values": col_data.nunique(),
                "missing_values": col_data.isna().sum()
            }
            
            # Add numerical stats if applicable
            if pd.api.types.is_numeric_dtype(col_data):
                result.update({
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std())
                })
            
            # Add categorical stats if applicable
            if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                # Get top 5 most common values
                value_counts = col_data.value_counts().head(5).to_dict()
                result["top_values"] = value_counts
            
            return result
        
        @Tool
        def run_query(data: Dict[str, Any], query: str) -> Dict[str, Any]:
            """Run a pandas query on the dataframe."""
            try:
                # Convert dict back to DataFrame for processing
                df = pd.DataFrame(data)
                
                # For safety, we'll limit to simple queries
                if "=" in query:
                    # Handle simple filtering
                    parts = query.split("=")
                    if len(parts) == 2:
                        col, val = parts[0].strip(), parts[1].strip()
                        if col in df.columns:
                            result = df[df[col] == val]
                            return {
                                "result": result.head(10).to_dict(orient="records"),
                                "count": len(result)
                            }
                
                # Default to returning summary
                return {
                    "error": "Complex queries not supported. Try asking a specific question instead."
                }
            except Exception as e:
                return {"error": str(e)}
        
        @Tool
        def analyze_correlation(data: Dict[str, Any], column1: str, column2: str) -> Dict[str, Any]:
            """Analyze correlation between two columns."""
            # Convert dict back to DataFrame for processing
            df = pd.DataFrame(data)
            
            if column1 not in df.columns or column2 not in df.columns:
                return {"error": f"One or both columns not found: {column1}, {column2}"}
            
            try:
                # Check if both columns are numeric
                if pd.api.types.is_numeric_dtype(df[column1]) and pd.api.types.is_numeric_dtype(df[column2]):
                    correlation = df[column1].corr(df[column2])
                    return {
                        "correlation": float(correlation),
                        "interpretation": self._interpret_correlation(correlation)
                    }
                else:
                    return {"error": "Correlation analysis requires numeric columns"}
            except Exception as e:
                return {"error": str(e)}
        
        # Create the agent with the tools
        # Fix: Use the correct model format for Ollama
        agent = Agent(
            model="ollama:llama3.1",  # Changed from ollama:llama3.1:8b-instruct-q8_0
            result_type=QueryResult,
            deps_type=CSVData,
            tools=[get_dataframe_info, get_column_stats, run_query, analyze_correlation],
            system_prompt="""
            You are a CSV data analysis assistant. Your job is to answer questions about CSV data.
            When answering:
            1. Be precise and accurate with numerical values
            2. Suggest visualizations when appropriate
            3. Explain your reasoning
            4. If you're unsure, admit it rather than making up information
            
            Use the provided tools to analyze the data and provide insights.
            """
        )
        
        return agent
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret a correlation coefficient."""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
            
        direction = "positive" if correlation >= 0 else "negative"
        
        return f"There is a {strength} {direction} correlation ({correlation:.2f}) between these variables."
    
    def process_query(self, df: pd.DataFrame, summary: Dict[str, Any], query: str) -> Tuple[QueryResult, Optional[str]]:
        """
        Process a user query about the CSV data.
        
        Args:
            df: The pandas DataFrame containing the CSV data
            summary: Summary information about the DataFrame
            query: The user's question
            
        Returns:
            Tuple[QueryResult, Optional[str]]: (result, error_message)
        """
        try:
            # Convert DataFrame to dict for Pydantic compatibility
            df_dict = df.to_dict(orient="list")
            
            # Create the CSV data object
            csv_data = CSVData(data=df_dict, summary=summary)
            
            # Run the query through the agent
            result = self.agent.run(query, deps=csv_data)
            
            return result, None
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Create a fallback result
            fallback_result = QueryResult(
                answer=f"I encountered an error while processing your query: {str(e)}",
                explanation="There was a technical issue. Please try rephrasing your question or check if the question is relevant to the uploaded data."
            )
            return fallback_result, str(e)
