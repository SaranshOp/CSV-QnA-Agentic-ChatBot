import gradio as gr
import pandas as pd
import numpy as np
import os
import io
import logging
from csv_handler import CSVHandler
from visualization import Visualizer
from llm_agent import CSVQueryAgent
from utils import safe_execute, format_error_message

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize components
csv_handler = CSVHandler()
visualizer = Visualizer()

# Initialize LLM agent with proper error handling
try:
    llm_agent = CSVQueryAgent()
    logger.info("Successfully initialized CSVQueryAgent")
except Exception as e:
    logger.error(f"Error initializing CSVQueryAgent: {str(e)}")
    # We'll handle this in the query processing function

# Global variables to store state
current_df = None
current_filename = "uploaded_file.csv"  # Default filename when actual name is not available

def process_csv_file(file_obj):
    """Process uploaded CSV file and return preview."""
    global current_df, current_filename
    
    if file_obj is None:
        return None, "Please upload a CSV file."
    
    # Store original filename if available
    if hasattr(file_obj, 'name'):
        current_filename = os.path.basename(file_obj.name)
    
    preview_df, error = csv_handler.process_csv(file_obj)
    if error:
        return None, error
    
    current_df = csv_handler.get_dataframe()
    summary = csv_handler.get_summary()
    
    # Create a summary message
    summary_msg = f"""
    CSV File: {current_filename}
    Rows: {summary['rows']}
    Columns: {summary['columns']}
    Column Names: {', '.join(summary['column_names'])}
    """
    
    return preview_df, summary_msg

def process_query(query):
    """Process a user query about the CSV data."""
    global current_df
    
    if current_df is None:
        return "Please upload a CSV file first.", None
    
    if not query or query.strip() == "":
        return "Please enter a question about the data.", None
    
    # Get the summary from the CSV handler
    summary = csv_handler.get_summary()
    
    try:
        # Check if LLM agent is available
        if 'llm_agent' not in globals() or llm_agent is None:
            return "LLM agent is not available. Please check your Ollama installation and model availability.", None
        
        # Process the query using the LLM agent
        result, error = llm_agent.process_query(current_df, summary, query)
        
        if error:
            return f"Error processing query: {error}", None
        
        # Check if we should generate a plot
        fig = None
        if result.plot_type and result.x_column:
            try:
                fig = visualizer.create_plot_figure(
                    df=current_df,
                    plot_type=result.plot_type,
                    x_column=result.x_column,
                    y_column=result.y_column,
                    title=result.title
                )
            except Exception as e:
                result.answer += f"\n\nNote: Could not generate plot: {str(e)}"
        
        # Format the response
        response = f"Answer: {result.answer}"
        if result.explanation:
            response += f"\n\nExplanation: {result.explanation}"
        
        return response, fig
    
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}", None

def create_demo_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="CSV Query Assistant") as demo:
        gr.Markdown("# CSV Query Assistant")
        gr.Markdown("Upload a CSV file and ask questions about its contents.")
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    type="binary"  # This returns bytes, not a file-like object
                )
            
        with gr.Row():
            with gr.Column():
                preview = gr.DataFrame(label="Data Preview")
                status_message = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Ask a question about your data",
                    placeholder="e.g., What is the average price? Show me a histogram of bedrooms."
                )
                query_button = gr.Button("Submit Query")
        
        with gr.Row():
            with gr.Column():
                response_output = gr.Textbox(label="Response", interactive=False)
                plot_output = gr.Plot(label="Visualization")
        
        # Add example queries
        gr.Examples(
            examples=[
                ["What is the average price in this dataset?"],
                ["Show me a histogram of the number of bedrooms."],
                ["What's the correlation between price and square footage?"],
                ["What are the top 5 most expensive properties?"],
                ["Summarize this dataset for me."]
            ],
            inputs=query_input
        )
        
        # Set up event handlers
        file_input.change(
            fn=process_csv_file,
            inputs=file_input,
            outputs=[preview, status_message]
        )
        
        query_button.click(
            fn=process_query,
            inputs=query_input,
            outputs=[response_output, plot_output]
        )
        
    return demo

# Create and launch the app
if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch()
