"""
rag_matcher.py
Semantic Policy Retrieval (RAG) for Custom Objectives.
Uses Vector Space Modeling with Character n-grams (char_wb) + Cosine Similarity.
Provides robust semantic retrieval that is resilient to spelling errors and typos
without requiring external LLM APIs.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.playbooks import PLAYBOOKS

class PolicyRAGMatcher:
    def __init__(self):
        self.playbook_names = list(PLAYBOOKS.keys())
        
        # Build enriched text corpus from playbook names, descriptions, and keywords
        self.corpus = [
            f"{name}. {PLAYBOOKS[name]['description']} {' '.join(PLAYBOOKS[name].get('keywords', []))}"
            for name in self.playbook_names
        ]
        
        # Use word-boundary character n-grams (3 to 5 chars) with sublinear TF.
        # This breaks words into sub-word tokens (e.g. 'predictive' -> 'pre', 'red', 'edi', 'dic', 'ict')
        # which guarantees typo-resilience and fuzzy matching across variations.
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def match(self, user_query: str) -> dict:
        """
        Embeds user query into character n-gram vector space,
        computes cosine similarity against all playbooks,
        and retrieves the optimal cleaning policy.
        """
        if not user_query or not user_query.strip():
            return {
                "matched_objective": "Machine learning readiness",
                "similarity_score": 100.0,
                "playbook": PLAYBOOKS["Machine learning readiness"]
            }

        # Embed query into the sub-word vector space
        query_vec = self.vectorizer.transform([user_query])
        
        # Calculate Cosine Similarity: cos(theta) = (A . B) / (||A|| * ||B||)
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])
        best_name = self.playbook_names[best_idx]

        # Fallback if query has near-zero similarity to any playbook
        if best_score < 0.04:
            best_name = "Machine learning readiness"
            best_score = 0.50

        return {
            "matched_objective": best_name,
            "similarity_score": round(best_score * 100, 1),
            "playbook": PLAYBOOKS[best_name]
        }
