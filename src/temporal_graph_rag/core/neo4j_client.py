"""
Neo4j client for semantic search operations.
"""

import numpy as np
import unicodedata
import re
from typing import List, Dict, Any
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util


class Neo4jClient:
    """Client for Neo4j database operations."""
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the database connection."""
        self.driver.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing accents and converting to ASCII."""
        nfkd_form = unicodedata.normalize('NFKD', text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        
    def extract_entities_llm(self, query: str, groq_client) -> str:
        """
        Extract entities from query using LLM.
        
        Args:
            query: Input query
            groq_client: Groq client for LLM calls
            
        Returns:
            Cleaned entity string
        """
        prompt = (
            "Extract only the most relevant entity or keyword from the following question.\n"
            "Return just the name(s), without any extra text or explanation.\n\n"
            f"Question: {query}\n\nEntities:"
        )
        raw_entities = groq_client.query(prompt).strip()

        # Clean output: remove (tags), bullets, quotes, and extra whitespace
        cleaned = re.sub(r"\s*\([^)]*\)", "", raw_entities)     # Remove (type)
        cleaned = re.sub(r"^\*+", "", cleaned)                  # Remove bullets like "* "
        cleaned = re.sub(r"[\"'`]", "", cleaned)                # Remove quotes
        cleaned = re.sub(r"\s+", " ", cleaned).strip()          # Normalize spaces
        return cleaned
        
    def semantic_search(self, query: str, embedder: SentenceTransformer, 
                       groq_client, similarity_threshold: float = 0.6) -> str:
        """
        Perform semantic search using Neo4j database.
        
        Args:
            query: Search query
            embedder: Sentence transformer for embeddings
            groq_client: Groq client for LLM operations
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Generated answer based on search results
        """
        entities = self.extract_entities_llm(query, groq_client)
        entities = [self.normalize_text(e.strip()) for e in entities.split(",") if e.strip()]
        
        # Generate embeddings for each entity
        entity_embeddings = [embedder.encode(e).tolist() for e in entities]
        
        # Fetch all candidate triplets with their embeddings
        with self.driver.session() as session:
            candidates = session.run(
                """
                MATCH (s)-[r]->(o)
                WHERE s.embedding IS NOT NULL AND o.embedding IS NOT NULL
                RETURN s.name AS subject, 
                       r.type AS predicate, 
                       o.name AS object,
                       s.embedding AS subj_emb,
                       o.embedding AS obj_emb,
                       r.date AS date
                """
            ).data()
        
        # Calculate similarities in Python
        all_triplets = set()
        for entity, emb in zip(entities, entity_embeddings):
            for record in candidates:
                try:
                    subj_sim = self.cosine_similarity(emb, record["subj_emb"])
                    obj_sim = self.cosine_similarity(emb, record["obj_emb"])
                    
                    if subj_sim > similarity_threshold or obj_sim > similarity_threshold:
                        triplet = f"({record['subject']}) -[{record['predicate']}]-> ({record['object']} Date:{record['date']})"
                        all_triplets.add(triplet)
                except Exception as e:
                    print(f"Error processing record: {str(e)}")
                    continue
        
        # Prepare context
        if not all_triplets:
            triplet_str = "No relevant triplets found."
        else:
            triplet_str = "\n".join(all_triplets)
        
        final_prompt = f"""Answer this query: {query}
        Using ONLY the following verified information from our knowledge graph:
        {triplet_str}
        
        If no relevant information exists, respond "I don't have verified information about this."
        """
        
        return groq_client.query(final_prompt)
