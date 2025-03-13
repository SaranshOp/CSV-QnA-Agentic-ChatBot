import os
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_execute(func, *args, default_return=None, **kwargs):
    """Execute a function safely with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default_return

def format_error_message(error: Exception) -> str:
    """Format an exception into a user-friendly error message."""
    error_type = type(error).__name__
    error_message = str(error)
    return f"Error ({error_type}): {error_message}"
