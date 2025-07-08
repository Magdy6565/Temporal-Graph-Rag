"""
Question processing and answering logic.
"""

import json
import re
import requests
from typing import Dict, List, Any, Optional
from .groq_client import GroqClient
from ..utils.date_utils import normalize_date


class QuestionProcessor:
    """Handles question classification, entity extraction, and answer generation."""
    
    def __init__(self, embedder, neo4j_client, groq_api_key: Optional[str] = None):
        """
        Initialize the question processor.
        
        Args:
            embedder: Sentence transformer for embeddings
            neo4j_client: Neo4j client for semantic search
            groq_api_key: Groq API key for LLM operations
        """
        self.embedder = embedder
        self.neo4j_client = neo4j_client
        self.groq_client = GroqClient(groq_api_key) if groq_api_key else None
        
    def classify_question(self, question: str) -> str:
        """
        Classify question type based on content.
        
        Args:
            question: Input question
            
        Returns:
            Question type: "Person", "Date", or "Other"
        """
        q = question.lower()

        # Person-focused questions
        if re.search(r"\bwho\b", q):
            return "Person"
        if re.search(r"\bwho (wrote|composed|created|painted|founded|led|directed)\b", q):
            return "Person"

        # Date-focused questions (more specific)
        if re.search(r"\b(on what (day|date|year)|what year|when|how long|how many|how much|what years)\b", q):
            return "Date"
        if re.match(r"^in \d{4}", q) and not re.search(r"\bwhat (incident|event|happened|caused)\b", q):
            return "Date"

        # Default fallback
        return "Other"
    
    def extract_entities_dates_predicate(self, question: str) -> Dict[str, Any]:
        """
        Extract entities, dates, and predicates from question using Groq.
        
        Args:
            question: Input question
            
        Returns:
            Dictionary with extracted entities, dates, and predicate
        """
        if not self.groq_client:
            raise ValueError("Groq API key not provided")
            
        system_prompt = """You are an expert at extracting structured information. Your goal is to populate a JSON object with 'entities', 'dates', and a 'predicate' based on the user's question.

Follow these rules precisely:
1.  **Identify Subjects for "entities"**:
    * Mentally break down the question to identify the main subject(s) of the core action or description. These subjects will be your "entities".
    * Subjects can be proper nouns (e.g., "Obama", "Paris"), or general noun phrases (e.g., "the man", "the red car", "companies", "the project manager").
    * If a question asks about "who" or "what" performed an action and the actor/subject is specified in the question, list that actor/subject.
    * If the subject is implicit (e.g., in "what happened?"), or if the question asks "who" or "what" and the subject is the answer being sought, the "entities" list should typically be empty.
2.  **Extract "dates"**:
    * Extract ALL date mentions in any format (e.g., "2nd March 2020", "May 2012", "2020", "yesterday", "last Monday", "first day", "Q2 2023").
    * Preserve original textual representation of dates, including ordinals (e.g., "2nd", "24th", "1st"). This should be a list of strings.
3.  **Determine "predicate"**:
    * Identify the main verb or action phrase that describes the central event or the relationship involving the primary subject(s). This will be the "predicate". This should be a single string.
    * For questions about roles or states (e.g., "who is the CEO of X"), the predicate might be a phrase like "is CEO of" or simply "is".
4.  **Output Format**:
    * Return a single, valid JSON object.
    * The JSON object must have exactly these three keys:
        * "entities": A list of strings. This field **must always be a list**, even if it contains only one subject (e.g., `["subject1"]`) or is empty (e.g., `[]`). Each string in the list should be a distinct, non-empty entity/subject.
        * "dates": A list of strings. Each string should be an extracted date phrase.
        * "predicate": A single string (or null if not clearly applicable).
"""

        user_prompt = f"""Extract from this question: entities, dates, and predicate.
Examples:
"what happened on 2nd March 2020?" → {{"entities": [], "dates": ["2nd March 2020"], "predicate": "happened"}}
"what did Obama say in May 2012?" → {{"entities": ["Obama"], "dates": ["May 2012"], "predicate": "say"}}
"What tasks did the project manager assign last Monday?" → {{"entities": ["the project manager"], "dates": ["last Monday"], "predicate": "assign"}}
"When was the old bridge closed for repairs?" → {{"entities": ["the old bridge"], "dates": ["When"], "predicate": "closed for repairs"}} 

Question: "{question}"
"""
        
        try:
            response = self.groq_client.query_json(system_prompt, user_prompt)
            extracted = json.loads(response) if isinstance(response, str) else response

            # Process entities
            final_entities = []
            raw_entities_value = extracted.get("entities")
            if isinstance(raw_entities_value, str):
                if raw_entities_value.strip():
                    final_entities.append(raw_entities_value.strip())
            elif isinstance(raw_entities_value, list):
                for item in raw_entities_value:
                    if isinstance(item, str) and item.strip():
                        final_entities.append(item.strip())

            # Process dates
            final_dates = []
            if "dates" in extracted and extracted["dates"]:
                for date_text in extracted.get("dates", []):
                    if date_text:
                        norm = normalize_date(str(date_text))
                        if norm:
                            final_dates.append(norm)
            
            if not final_dates:
                final_dates.append({"normalized_date": None, "type": 4, "original": None})

            return {
                "entities": final_entities,
                "dates": final_dates,
                "predicate": extracted.get("predicate", None)
            }
            
        except Exception as e:
            return {
                "entities": [],
                "dates": [{"normalized_date": None, "type": 4, "original": None}],
                "predicate": None
            }
    
    def generate_answer_from_triplet(self, user_question: str, triplet: Dict[str, Any]) -> str:
        """
        Generate natural language answer from triplet result.
        
        Args:
            user_question: Original user question
            triplet: Triplet result dictionary
            
        Returns:
            Generated natural language answer
        """
        if not self.groq_client:
            raise ValueError("Groq API key not provided")
            
        subject = triplet['subject']
        predicate = triplet['predicate']
        obj = triplet['object']

        prompt = f"""
Using the information:

Subject: {subject}
Predicate: {predicate}
Object: {obj}

Write a very short natural sentence. Do NOT add explanations, do NOT say you are sorry. Only say the main information.

Example format:
"[subject] [predicate] [object]"

Answer:
"""
        
        return self.groq_client.query(prompt)
    
    def answer_question(self, question: str, temporal_graph) -> str:
        """
        Answer a question using the temporal graph and Neo4j.
        
        Args:
            question: Natural language question
            temporal_graph: TemporalGraphRAG instance
            
        Returns:
            Generated answer
        """
        # Step 1: Extract structured info
        question_type = self.classify_question(question)
        extracted = self.extract_entities_dates_predicate(question)
        
        if not extracted["entities"] or not extracted["dates"]:
            return "Sorry, I couldn't understand the question."

        subject = extracted["entities"][0]
        predicate = extracted["predicate"]
        date_info = extracted["dates"][0]
        
        # Step 2: Determine search strategy based on date type
        if date_info['type'] == 0:  # Full date MM/DD/YYYY
            triplets = temporal_graph.search(
                date_info["normalized_date"], subject, predicate, question_type
            )
            if not triplets:
                return "Sorry, I couldn't find the answer in the news data."
            triplet = triplets[0]
            
        elif date_info['type'] == 1:  # Month/Year
            date = date_info['normalized_date']
            month, _, year = date.split('/')
            month, year = int(month), int(year)
            triplets = temporal_graph.search_month(
                month, year, subject, predicate, question_type
            )
            if not triplets:
                return "Sorry, I couldn't find the answer in the news data."
            triplet = triplets[0]
            
        elif date_info['type'] == 3:  # Year only
            return self.neo4j_client.semantic_search(
                question, self.embedder, self.groq_client
            )
            
        else:  # No date or unclear date
            return self.neo4j_client.semantic_search(
                question, self.embedder, self.groq_client
            )
            
        # Step 3: Generate final answer
        return self.generate_answer_from_triplet(question, triplet)
