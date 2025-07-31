"""
Basic RAG Scorers for NovaEval.

This module contains fundamental scorers for RAG evaluation including:
- Basic RAG scorers (from rag.py)
- Advanced retrieval scorers (precision, recall, ranking, diversity)
- Semantic similarity scorers
- Aggregate scoring
"""
import re
import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score
from sentence_transformers import SentenceTransformer

# LangChain detection
try:
    import langchain
    from langchain_core.language_models.base import BaseLanguageModel
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseLanguageModel = None

from typing import Any, Optional, Union, List, TYPE_CHECKING
from novaeval.scorers.base import BaseScorer, ScoreResult
from novaeval.scorers.rag import (
    AnswerRelevancyScorer,
    FaithfulnessScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    RAGASScorer
)

# Forward reference for type hints
if TYPE_CHECKING:
    from typing_extensions import TypedDict


def _call_llm(model, prompt: str):
    """
    Helper to call the LLM, supporting both string (OpenAI) and LangChain LLM objects.
    """
    if _HAS_LANGCHAIN and BaseLanguageModel and isinstance(model, BaseLanguageModel):
        # LangChain LLM: use .invoke or .generate
        if hasattr(model, "invoke"):
            return model.invoke(prompt)
        elif hasattr(model, "generate"):
            return model.generate([prompt]).generations[0][0].text
        else:
            raise ValueError("LangChain LLM does not support invoke or generate.")
    elif isinstance(model, str):
        # Built-in: string model name, use OpenAI API
        raise NotImplementedError("String-based LLM calling not implemented. Please provide a LangChain LLM or implement OpenAI call here.")
    else:
        # Assume model has a .generate method (NovaEval style)
        return model.generate(prompt)


class ContextualPrecisionScorerPP(BaseScorer):
    """
    Enhanced contextual precision scorer with ranking awareness.
    Evaluates how relevant the retrieved context is to the query.
    """
    
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="ContextualPrecisionScorerPP", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not input_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or input provided", metadata={})
        
        prompt = f"""
        Evaluate the relevance of the retrieved context to the query.
        
        Query: {input_text}
        Retrieved Context: {context}
        
        Rate the relevance from 0-10 where:
        0: Completely irrelevant
        5: Somewhat relevant
        10: Highly relevant and directly addresses the query
        
        Provide only the numerical score (0-10):
        """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_relevance_score(response)
            passed = score >= self.threshold * 10
            
            return ScoreResult(
                score=score / 10.0,  # Normalize to 0-1
                passed=passed,
                reasoning=f"Context relevance score: {score}/10",
                metadata={"raw_score": score, "threshold": self.threshold * 10}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Evaluation failed: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        # Async wrapper for _call_llm
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call_llm, self.model, prompt)
    
    def _parse_relevance_score(self, resp):
        # Extract numerical score from response
        numbers = re.findall(r'\b(?:10|[0-9])\b', resp)
        if numbers:
            return float(numbers[-1])
        return 5.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0


class ContextualRecallScorerPP(BaseScorer):
    """
    Enhanced contextual recall scorer with comprehensive coverage analysis.
    Evaluates how much of the relevant information is captured in the retrieved context.
    """
    
    def __init__(self, model, threshold=0.7, top_k=5, **kwargs):
        super().__init__(name="ContextualRecallScorerPP", **kwargs)
        self.threshold = threshold
        self.model = model
        self.top_k = top_k
    
    async def evaluate(self, input_text, output_text, expected_output=None, context=None, **kwargs: Any) -> ScoreResult:
        if not context or not input_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or input provided", metadata={})
        
        # Extract key information points from the query
        claims = self._parse_claims(input_text)
        
        if not claims:
            return ScoreResult(score=0.0, passed=False, reasoning="No claims found in input", metadata={})
        
        prompt = f"""
        Evaluate how well the retrieved context covers the key information points from the query.
        
        Query: {input_text}
        Key Information Points: {', '.join(claims)}
        Retrieved Context: {context}
        
        For each information point, rate coverage from 0-10:
        0: Not covered at all
        5: Partially covered
        10: Fully covered
        
        Provide the average coverage score (0-10):
        """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_relevance_score(response)
            passed = score >= self.threshold * 10
            
            return ScoreResult(
                score=score / 10.0,  # Normalize to 0-1
                passed=passed,
                reasoning=f"Context recall score: {score}/10",
                metadata={"raw_score": score, "claims": claims, "threshold": self.threshold * 10}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Evaluation failed: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call_llm, self.model, prompt)
    
    def _parse_claims(self, text):
        # Simple claim extraction - can be enhanced
        sentences = text.split('.')
        return [s.strip() for s in sentences if len(s.strip()) > 10][:self.top_k]
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0


class ContextualF1Scorer(BaseScorer):
    """
    F1 score combining precision and recall for contextual evaluation.
    """
    
    def __init__(self, precision_scorer, recall_scorer, threshold=0.5, **kwargs):
        super().__init__(name="ContextualF1Scorer", **kwargs)
        self.precision_scorer = precision_scorer
        self.recall_scorer = recall_scorer
        self.threshold = threshold
    
    def score(self, prediction, ground_truth, context=None):
        # Calls the precision and recall scorers, then computes F1
        precision = self.precision_scorer.score(prediction, ground_truth, context)
        recall = self.recall_scorer.score(prediction, ground_truth, context)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        passed = f1_score >= self.threshold
        
        return ScoreResult(
            score=f1_score,
            passed=passed,
            reasoning=f"F1 Score: {f1_score:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})",
            metadata={"precision": precision, "recall": recall}
        )


class RetrievalRankingScorer(BaseScorer):
    """
    Computes ranking metrics for retrieved context.
    """
    
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(name="RetrievalRankingScorer", **kwargs)
        self.threshold = threshold
    
    def score(self, prediction, ground_truth, context=None):
        # Computes NDCG, MAP, and MRR for the retrieved context
        if not context or "rankings" not in context:
            return ScoreResult(score=0.0, passed=False, reasoning="No ranking data provided", metadata={})
        
        rankings = context["rankings"]
        relevance_scores = context.get("relevance_scores", [1.0] * len(rankings))
        
        try:
            # NDCG
            ndcg = ndcg_score([relevance_scores], [rankings], k=len(rankings))
            
            # MAP
            map_score = average_precision_score(relevance_scores, rankings)
            
            # MRR
            mrr = 0.0
            for i, rank in enumerate(rankings):
                if relevance_scores[i] > 0:
                    mrr = 1.0 / (rank + 1)
                    break
            
            avg_score = (ndcg + map_score + mrr) / 3.0
            passed = avg_score >= self.threshold
            
            return ScoreResult(
                score=avg_score,
                passed=passed,
                reasoning=f"Ranking Score: {avg_score:.3f} (NDCG: {ndcg:.3f}, MAP: {map_score:.3f}, MRR: {mrr:.3f})",
                metadata={"ndcg": ndcg, "map": map_score, "mrr": mrr}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Ranking computation failed: {str(e)}", metadata={})


class SemanticSimilarityScorer(BaseScorer):
    """
    Computes semantic similarity between query and retrieved context.
    """
    
    def __init__(self, threshold=0.7, embedding_model='all-MiniLM-L6-v2', **kwargs):
        super().__init__(name="SemanticSimilarityScorer", **kwargs)
        self.threshold = threshold
        self.embedding_model = embedding_model
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.embedding_model)
    
    def score(self, prediction, ground_truth, context=None):
        # Embeds the query and all chunks, computes mean similarity and diversity
        if not context or not ground_truth:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or query provided", metadata={})
        
        try:
            self._load_model()
            
            query = ground_truth
            chunks = context.get("chunks", [context.get("context", "")])
            
            # Compute embeddings
            query_embedding = self.model.encode([query])[0]
            chunk_embeddings = self.model.encode(chunks)
            
            # Compute similarities
            similarities = []
            for chunk_emb in chunk_embeddings:
                sim = np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
                similarities.append(sim)
            
            mean_similarity = np.mean(similarities)
            diversity = 1.0 - np.std(similarities)  # Higher std = lower diversity
            
            avg_score = (mean_similarity + diversity) / 2.0
            passed = avg_score >= self.threshold
            
            return ScoreResult(
                score=avg_score,
                passed=passed,
                reasoning=f"Semantic Score: {avg_score:.3f} (Mean Similarity: {mean_similarity:.3f}, Diversity: {diversity:.3f})",
                metadata={"mean_similarity": mean_similarity, "diversity": diversity, "similarities": similarities}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Semantic similarity computation failed: {str(e)}", metadata={})


class RetrievalDiversityScorer(BaseScorer):
    """
    Evaluates diversity of retrieved context chunks.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="RetrievalDiversityScorer", **kwargs)
    
    def score(self, prediction, ground_truth, context=None):
        if not context or "chunks" not in context:
            return ScoreResult(score=0.0, passed=False, reasoning="No chunks provided", metadata={})
        
        chunks = context["chunks"]
        
        if len(chunks) <= 1:
            return ScoreResult(score=0.0, passed=False, reasoning="Insufficient chunks for diversity calculation", metadata={})
        
        try:
            # Simple diversity based on unique content
            unique_chunks = set(chunks)
            diversity_score = len(unique_chunks) / len(chunks)
            
            return ScoreResult(
                score=diversity_score,
                passed=diversity_score > 0.5,
                reasoning=f"Diversity Score: {diversity_score:.3f} ({len(unique_chunks)} unique out of {len(chunks)} chunks)",
                metadata={"unique_chunks": len(unique_chunks), "total_chunks": len(chunks)}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Diversity computation failed: {str(e)}", metadata={})


class AggregateRetrievalScorer(BaseScorer):
    """
    Combines multiple retrieval scorers with weighted averaging.
    """
    
    def __init__(self, scorers: dict, weights: Optional[dict] = None, **kwargs):
        super().__init__(name="AggregateRetrievalScorer", **kwargs)
        self.scorers = scorers
        self.weights = weights or {name: 1.0 for name in scorers.keys()}
    
    def score(self, prediction, ground_truth, context=None):
        # Calls each scorer, extracts main score, and computes weighted average
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, scorer in self.scorers.items():
            try:
                result = scorer.score(prediction, ground_truth, context)
                score = result.score if hasattr(result, 'score') else result
                weight = self.weights.get(name, 1.0)
                
                scores[name] = score
                weighted_sum += score * weight
                total_weight += weight
            except Exception as e:
                scores[name] = 0.0
                print(f"Warning: Scorer {name} failed: {e}")
        
        if total_weight == 0:
            return ScoreResult(score=0.0, passed=False, reasoning="All scorers failed", metadata=scores)
        
        final_score = weighted_sum / total_weight
        passed = final_score >= 0.5  # Default threshold
        
        return ScoreResult(
            score=final_score,
            passed=passed,
            reasoning=f"Aggregate Score: {final_score:.3f}",
            metadata={"individual_scores": scores, "weights": self.weights}
        ) 