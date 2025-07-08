"""
Text processing utilities.
"""

import re
from typing import List, Dict


# Regex patterns for filtering
DATE_REGEX = r"\b(\d{1,2}[ \-/])?(January|February|March|April|May|June|July|August|September|October|November|December)[ \-/]\d{2,4}\b|\b\d{4}\b|\b\d{1,4}[ ]?B\.?C\.?\b"
NUMERIC_REGEX = r"\b\d+\b|\bmillion\b|\bbillion\b|\bthousand\b|\bhundred\b|\bfew\b|\bseveral\b|\bone\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b|\bsix\b|\bseven\b|\beight\b|\bnine\b|\bten\b"
COMBINED_REGEX = f"({DATE_REGEX})|({NUMERIC_REGEX})"


def filter_triplets_with_dates_or_numbers(triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filter triplets that contain dates or numbers in subject or object.
    
    Args:
        triplets: List of triplet dictionaries
        
    Returns:
        Filtered list of triplets containing dates or numbers
    """
    filtered = []
    for triplet in triplets:
        subj = triplet["subject"].lower()
        obj = triplet["object"].lower()
        if re.search(COMBINED_REGEX, subj, re.IGNORECASE) or re.search(COMBINED_REGEX, obj, re.IGNORECASE):
            filtered.append(triplet)
    return filtered


def filter_triplets_with_names_or_locations(triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filter triplets that likely contain person names or locations.
    
    Args:
        triplets: List of triplet dictionaries
        
    Returns:
        Filtered list of triplets containing names or locations
    """
    # Simple heuristic: look for capitalized words that might be names
    name_patterns = [
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last name pattern
        r'\b(Mr|Mrs|Dr|Prof|President|Minister|King|Queen)\s+[A-Z][a-z]+\b',  # Titles
        r'\b[A-Z][a-z]+\s+(City|State|Country|Province|Region)\b',  # Locations
    ]
    
    combined_pattern = '|'.join(name_patterns)
    
    filtered = []
    for triplet in triplets:
        subj = triplet["subject"]
        obj = triplet["object"]
        if (re.search(combined_pattern, subj) or re.search(combined_pattern, obj) or
            any(word[0].isupper() for word in subj.split()) or
            any(word[0].isupper() for word in obj.split())):
            filtered.append(triplet)
    return filtered


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', text)
    
    return text


def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    # Simple extraction based on capitalized words and noun phrases
    phrases = []
    
    # Extract capitalized phrases (likely proper nouns)
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    phrases.extend(capitalized_phrases[:max_phrases//2])
    
    # Extract other significant words (longer than 3 characters)
    words = re.findall(r'\b\w{4,}\b', text.lower())
    significant_words = [word for word in words if word not in ['this', 'that', 'with', 'from', 'they', 'were', 'been', 'have']]
    phrases.extend(significant_words[:max_phrases - len(phrases)])
    
    return phrases[:max_phrases]
