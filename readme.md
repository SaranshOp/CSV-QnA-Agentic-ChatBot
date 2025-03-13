# CSV Question Answering and Visualization Application

This project is a Gradio-based application that allows users to upload CSV files and ask questions about the data. The application processes the CSV data and provides answers using a language model. It also supports generating visualizations based on the data.

## Features

- **CSV File Upload**: Users can upload CSV files for analysis.
- **Question Answering**: Users can ask questions about the uploaded CSV data.
- **Data Visualization**: The application can generate various types of plots (e.g., line, bar, scatter, histogram) based on the data.
- **Error Handling**: The application includes robust error handling to manage issues during CSV processing and query execution.

## Components

- **Gradio Interface**: Provides a user-friendly interface for uploading files and submitting queries.
- **CSV Handler**: Validates and processes the uploaded CSV files.
- **LLM Agent**: Uses a language model to process queries and generate responses.
- **Visualizer**: Generates plots based on the data and query results.
- **Utilities**: Includes helper functions for safe execution and error formatting.

## Getting Started

1. **Install Dependencies**: Install the required Python packages using the `requirements.txt` file.
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the Application**: Launch the Gradio application.
   ```sh
   python app.py
   ```
3. **Upload CSV and Ask Questions**: Use the Gradio interface to upload a CSV file and ask questions about the data.
