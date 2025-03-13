import pandas as pd
import numpy as np
import io
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from utils import safe_execute, format_error_message

logger = logging.getLogger(__name__)

class CSVHandler:
    def __init__(self):
        self.df = None
        self.file_path = None
        self.columns = []
        self.column_types = {}
        self.summary = {}

    def validate_csv(self, file_obj) -> Tuple[bool, Optional[str]]:
        """
        Validate if the uploaded file is a proper CSV file.
        
        Args:
            file_obj: The uploaded file object
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Handle case where file_obj is bytes
            if isinstance(file_obj, bytes):
                # Create a BytesIO object
                file_data = io.BytesIO(file_obj)
                # Try to read as CSV
                df = pd.read_csv(file_data)
                # Store filename as None since we don't have it
                filename = None
            else:
                # Check file extension if name attribute exists
                if hasattr(file_obj, 'name') and not file_obj.name.lower().endswith('.csv'):
                    return False, "File must be a CSV file."
                
                # Store filename if available
                filename = getattr(file_obj, 'name', None)
                
                # Try to read the file as CSV
                if hasattr(file_obj, 'read'):
                    content = file_obj.read()
                    df = pd.read_csv(io.BytesIO(content))
                    # Reset file pointer if possible
                    if hasattr(file_obj, 'seek'):
                        file_obj.seek(0)
                else:
                    return False, "Invalid file object: missing read method"
            
            # Check if file has content
            if df.empty:
                return False, "CSV file is empty."
                
            # Check if file has at least one column
            if len(df.columns) == 0:
                return False, "CSV file has no columns."
            
            # Store validated data for later use
            self._validated_df = df
            self._validated_filename = filename
                
            return True, None
            
        except Exception as e:
            logger.error(f"CSV validation error: {str(e)}")
            return False, f"Invalid CSV file: {str(e)}"

    def process_csv(self, file_obj) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Process the uploaded CSV file.
        
        Args:
            file_obj: The uploaded file object
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (dataframe, error_message)
        """
        # Validate CSV first
        is_valid, error_message = self.validate_csv(file_obj)
        if not is_valid:
            return None, error_message
        
        try:
            # Use the dataframe we already created during validation
            self.df = self._validated_df
            self.file_path = self._validated_filename
            
            # Analyze columns
            self.columns = list(self.df.columns)
            self.column_types = {col: str(self.df[col].dtype) for col in self.columns}
            
            # Generate basic summary
            self.summary = {
                'rows': len(self.df),
                'columns': len(self.columns),
                'column_names': self.columns,
                'column_types': self.column_types,
                'missing_values': self.df.isna().sum().to_dict()
            }
            
            # Return the first 10 rows for preview
            return self.df.head(10), None
            
        except Exception as e:
            logger.error(f"CSV processing error: {str(e)}")
            return None, f"Error processing CSV: {str(e)}"
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the current dataframe."""
        return self.df
    
    def get_summary(self) -> Dict[str, Any]:
        """Return the summary of the CSV file."""
        return self.summary
    
    def get_column_info(self) -> Dict[str, str]:
        """Return information about columns."""
        return self.column_types
