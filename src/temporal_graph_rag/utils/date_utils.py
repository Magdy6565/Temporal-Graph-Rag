"""
Date utilities for temporal processing.
"""

import re
from datetime import datetime
from dateutil import parser
from typing import Dict, Any, Optional


def normalize_date(date_text: str) -> Optional[Dict[str, Any]]:
    """
    Normalize various date formats to MM/DD/YYYY.
    
    Args:
        date_text: Input date string in various formats
        
    Returns:
        Dictionary with normalized date info or None if parsing fails
        
    Types:
        0: Full date (MM/DD/YYYY)
        1: Month/Year (MM/01/YYYY) 
        3: Year only (01/01/YYYY)
        4: Failed to parse
        5: Error during parsing
    """
    try:
        original_text_for_output = date_text.strip()
        
        # Remove ordinal suffixes (1st, 2nd, 3rd, 4th)
        cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_text.strip())
        
        dt = None
        try:
            dt = parser.parse(cleaned, fuzzy=True, dayfirst=False, default=datetime(1900, 1, 1))
        except Exception:
            dt = parser.parse(cleaned, fuzzy=True, default=datetime(1900, 1, 1))
        
        # Determine date type and format accordingly
        if re.fullmatch(r'\d{4}', cleaned):  # Year only
            return {
                "normalized_date": f"01/01/{cleaned}", 
                "type": 3, 
                "original": original_text_for_output
            }
        elif re.fullmatch(
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*\s+\d{4}', 
            cleaned, re.IGNORECASE
        ):  # Month Year
            month_year = parser.parse(cleaned, default=datetime(1900, 1, 1))
            return {
                "normalized_date": month_year.strftime("%m/01/%Y"), 
                "type": 1, 
                "original": original_text_for_output
            }
        else:  # Full date
            return {
                "normalized_date": dt.strftime("%m/%d/%Y"), 
                "type": 0, 
                "original": original_text_for_output
            }
            
    except Exception as e:
        return {
            "normalized_date": None, 
            "type": 5, 
            "original": date_text, 
            "error": str(e)
        }


def date_to_value(date: str) -> int:
    """
    Convert date string to numerical value for hashing.
    
    Args:
        date: Date string in MM/DD/YYYY format
        
    Returns:
        Numerical value representing the date
    """
    date_obj = datetime.strptime(date, "%m/%d/%Y")
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    return year * 372 + month * 31 + day


def value_to_date(value: int) -> str:
    """
    Convert numerical value back to date string.
    
    Args:
        value: Numerical value representing date
        
    Returns:
        Date string in MM/DD/YYYY format
    """
    year = value // 372
    remainder = value % 372
    month = remainder // 31
    day = remainder % 31
    
    # Handle edge cases
    if month == 0:
        month = 1
    if day == 0:
        day = 1
        
    return f"{month:02d}/{day:02d}/{year}"
