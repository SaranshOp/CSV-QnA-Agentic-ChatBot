import pandas as pd
import gradio as gr
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
import tempfile
import traceback

# Define our data models
class CSVData(BaseModel):
    """CSV data for analysis"""
    filepath: str
    description: str = "A CSV file containing data for analysis"

class ColumnStats(BaseModel):
    """Statistics for a column in the dataset"""
    column_name: str
    data_type: str
    mean: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    unique_values: Optional[int] = None
    
class QueryResult(BaseModel):
    """Result of a query on CSV data"""
    answer: str = Field(description="The answer to the user's query")
    relevant_columns: Optional[List[str]] = Field(default_factory=list, description="Columns relevant to the query")
    statistics: Optional[List[ColumnStats]] = Field(default=None, description="Statistical information about relevant columns")
    error: Optional[str] = Field(default=None, description="Error message if query processing failed")

# Configure Ollama to work with Pydantic AI using OpenAI-compatible interface
ollama_model = OpenAIModel(
    model_name='llama3.1:latest',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
    system="You are a helpful CSV data analysis assistant. Your job is to answer questions about CSV data by analyzing it and providing accurate statistics and insights. Always explain your reasoning clearly. Always return relevant_columns as a list, even if empty."
)

# Create our agent
agent = Agent(
    ollama_model,
)

# Define tools for the agent
@agent.tool
def get_column_names(ctx: RunContext[CSVData], filepath: str) -> List[str]:
    """Get the column names from a CSV file"""
    try:
        df = pd.read_csv(filepath)
        return df.columns.tolist()
    except Exception as e:
        return [f"Error reading CSV: {str(e)}"]

@agent.tool
def get_column_stats(ctx: RunContext[CSVData], filepath: str, column_name: str) -> Dict[str, Any]:
    """Get statistics for a specific column"""
    try:
        df = pd.read_csv(filepath)
        if column_name not in df.columns:
            return {"error": f"Column '{column_name}' not found in the dataset"}
        
        col_data = df[column_name]
        stats = {
            "column_name": column_name,
            "data_type": str(col_data.dtype)
        }
        
        # Add numerical stats if applicable
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "min": float(col_data.min()),
                "max": float(col_data.max())
            })
        
        # Add categorical stats
        stats["unique_values"] = len(col_data.unique())
        
        return stats
    except Exception as e:
        return {"error": f"Error analyzing column: {str(e)}"}

@agent.tool
def filter_data(ctx: RunContext[CSVData], filepath: str, column: str, condition: str, value: Any) -> str:
    """Filter data based on a condition and return summary"""
    try:
        df = pd.read_csv(filepath)
        if column not in df.columns:
            return f"Column '{column}' not found in the dataset"
        
        if condition == "equals":
            filtered = df[df[column] == value]
        elif condition == "greater_than":
            filtered = df[df[column] > value]
        elif condition == "less_than":
            filtered = df[df[column] < value]
        elif condition == "contains":
            filtered = df[df[column].astype(str).str.contains(str(value))]
        else:
            return f"Unsupported condition: {condition}"
        
        return f"Filtered data has {len(filtered)} rows. First few rows:\n{filtered.head(3).to_string()}"
    except Exception as e:
        return f"Error filtering data: {str(e)}"

# Function to process queries
def process_query(csv_file, query):
    if csv_file is None:
        return "Please upload a CSV file first."
    
    try:
        # Create a temporary file to save the uploaded CSV
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "temp_upload.csv")
        
        # Handle the file upload correctly based on Gradio's file format
        if hasattr(csv_file, 'name'):  # For Gradio versions that return a file-like object
            # Copy the file content instead of just using the name
            with open(csv_file.name, 'rb') as src, open(temp_path, 'wb') as dst:
                dst.write(src.read())
        else:
            # For older Gradio versions or different return types
            if isinstance(csv_file, tuple) and len(csv_file) == 2:
                with open(temp_path, "wb") as f:
                    f.write(csv_file[1])
            else:
                # Try direct save
                with open(temp_path, "wb") as f:
                    if hasattr(csv_file, 'read'):  # If it's a file-like object
                        f.write(csv_file.read())
                    else:  # If it's bytes or similar
                        f.write(csv_file)
        
        # Verify the file was written correctly
        if os.path.getsize(temp_path) == 0:
            return "Error: The uploaded CSV file appears to be empty."
            
        # Try to read the file with pandas to verify it's valid
        try:
            test_df = pd.read_csv(temp_path)
            if len(test_df) == 0:
                return "Error: The CSV file contains no data rows."
            print(f"CSV file loaded successfully with {len(test_df)} rows and {len(test_df.columns)} columns.")
        except Exception as e:
            return f"Error: Could not parse the CSV file: {str(e)}"
        
        # Create CSV data object
        csv_data = CSVData(filepath=temp_path, description="Uploaded CSV file")
        
        try:
            # Process the query using our agent with increased retries
            result = agent.run_sync(
                query,
                result_type=QueryResult,
                deps={"csv_data": csv_data},
                model_settings={"max_retries": 3}
            )
            
            # Format the response
            response = f"**Answer:** {result.data.answer}\n\n"
            
            if result.data.relevant_columns:
                response += f"**Relevant Columns:** {', '.join(result.data.relevant_columns)}\n\n"
            
            if result.data.statistics:
                response += "**Statistics:**\n"
                for stat in result.data.statistics:
                    response += f"- {stat.column_name} ({stat.data_type}):\n"
                    if stat.mean is not None:
                        response += f"  - Mean: {stat.mean}\n"
                    if stat.median is not None:
                        response += f"  - Median: {stat.median}\n"
                    if stat.min is not None:
                        response += f"  - Min: {stat.min}\n"
                    if stat.max is not None:
                        response += f"  - Max: {stat.max}\n"
                    if stat.unique_values is not None:
                        response += f"  - Unique Values: {stat.unique_values}\n"
            
            if result.data.error:
                response += f"\n**Error:** {result.data.error}"
                
            return response
            
        except Exception as e:
            # Fallback to simple query without structured output
            print(f"Structured output failed: {str(e)}")
            print("Falling back to simple query...")
            
            simple_result = agent.run_sync(
                f"Answer this question about the CSV file at {temp_path}: {query}",
                deps={"csv_data": csv_data}
            )
            
            return f"**Answer:** {simple_result.data}"
            
    except Exception as e:
        return f"Error processing query: {str(e)}\n\n{traceback.format_exc()}"

# Create Gradio interface
with gr.Blocks(title="CSV Analyzer with Ollama & Pydantic AI") as demo:
    gr.Markdown("# CSV Data Analysis with Ollama & Pydantic AI")
    gr.Markdown("Upload a CSV file and ask questions about the data.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV File")
            query_input = gr.Textbox(label="Your Question", placeholder="What's the average value of column X?")
            submit_btn = gr.Button("Analyze")
        
        with gr.Column():
            output = gr.Markdown(label="Analysis Result")
    
    submit_btn.click(
        process_query,
        inputs=[file_input, query_input],
        outputs=output
    )
    
    # Create a sample CSV file for the examples
    sample_csv_path = os.path.join(tempfile.gettempdir(), "sample.csv")
    with open(sample_csv_path, "w", newline='') as f:
        f.write("""name,age,city,salary
John,34,New York,75000
Alice,29,San Francisco,120000
Bob,45,Chicago,82000
Maria,38,Boston,95000
David,41,Seattle,110000""")
    
    gr.Examples(
        examples=[
            [sample_csv_path, "What columns are in this dataset?"],
            [sample_csv_path, "What is the average age?"],
            [sample_csv_path, "Who has the highest salary?"],
            [sample_csv_path, "Show me data where salary is greater than 100000"]
        ],
        inputs=[file_input, query_input]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
