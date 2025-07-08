"""
Groq API client for LLM operations.
"""

import json
import requests
from typing import Optional, Dict, Any


class GroqClient:
    """Client for Groq API operations."""
    
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def query(self, prompt: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
        """
        Send a simple query to Groq API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                return data["choices"][0]["message"]["content"]
            else:
                return str(data)
                
        except Exception as e:
            return f"Error in Groq API call: {str(e)}"
    
    def query_json(self, system_prompt: str, user_prompt: str, 
                   temperature: float = 0, max_tokens: int = 400) -> str:
        """
        Send a query expecting JSON response.
        
        Args:
            system_prompt: System prompt for context
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            JSON response as string
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",  # Use faster model for JSON tasks
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Try to parse as JSON first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback: extract JSON from text
                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
                else:
                    raise ValueError("No valid JSON found in response")
                    
        except Exception as e:
            return json.dumps({
                "entities": [],
                "dates": [],
                "predicate": None,
                "error": str(e)
            })
