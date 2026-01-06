"""Hypothesis-based causal consistency classification using LLM scoring."""
from openai import OpenAI
from typing import List, Dict, Tuple
import re
import json


class BackstoryClassifier:
    """Classify backstory causal consistency using hypothesis scoring."""
    
    # 4-class label definitions
    LABELS = {
        'consistent': 1,      # Backstory is logically consistent with novel
        'inconsistent': 0,    # Backstory contradicts novel evidence
        'insufficient': 0,    # Not enough evidence to verify
        'contradictory': 0    # Direct contradictions found
    }
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", prompt_template: str = ""):
        """
        Initialize classifier.
        
        Args:
            api_key: OpenAI API key
            model: Model name
            prompt_template: Not used in new design (kept for compatibility)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def _check_evidence_sufficiency(self, evidence_chunks: List[Dict]) -> Tuple[bool, str]:
        """
        Check if retrieved evidence is sufficient for classification.
        
        Args:
            evidence_chunks: Retrieved evidence chunks
            
        Returns:
            (is_sufficient, reason)
        """
        if not evidence_chunks:
            return False, "No evidence chunks retrieved"
        
        if len(evidence_chunks) < 3:
            return False, f"Only {len(evidence_chunks)} chunks retrieved, need at least 3"
        
        # Check total evidence length
        total_chars = sum(len(chunk.get('text', '')) for chunk in evidence_chunks)
        if total_chars < 500:
            return False, f"Evidence too short ({total_chars} chars), need at least 500"
        
        return True, "Sufficient evidence available"
    
    def _score_hypotheses(self, backstory: str, evidence_chunks: List[Dict]) -> Dict[str, float]:
        """
        Use LLM to score each hypothesis.
        
        Args:
            backstory: Backstory text
            evidence_chunks: Retrieved evidence chunks
            
        Returns:
            Dict mapping label to score (0.0-1.0)
        """
        # Combine evidence
        evidence_text = "\n\n---\n\n".join([
            f"Chunk {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(evidence_chunks)
        ])
        
        # Scoring prompt
        prompt = f"""Analyze the logical and causal consistency between a backstory and novel evidence.

**Novel Evidence:**
{evidence_text}

**Backstory to Verify:**
{backstory}

**Task:** Score each hypothesis from 0.0 (definitely false) to 1.0 (definitely true).

Rate these hypotheses:
1. CONSISTENT: The backstory is logically and causally consistent with the evidence
2. INCONSISTENT: The backstory has logical flaws or timing issues (but no direct contradictions)
3. INSUFFICIENT: The evidence does not contain enough information to verify the backstory
4. CONTRADICTORY: The backstory directly contradicts facts stated in the evidence

Output ONLY a JSON object with scores:
{{"consistent": 0.0-1.0, "inconsistent": 0.0-1.0, "insufficient": 0.0-1.0, "contradictory": 0.0-1.0}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a literary analysis expert. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Parse JSON scores
            scores = json.loads(answer)
            
            # Validate and normalize
            valid_scores = {}
            for label in ['consistent', 'inconsistent', 'insufficient', 'contradictory']:
                score = float(scores.get(label, 0.0))
                valid_scores[label] = max(0.0, min(1.0, score))
            
            return valid_scores
            
        except Exception as e:
            print(f"Error scoring hypotheses: {e}")
            # Default to insufficient evidence on error
            return {
                'consistent': 0.0,
                'inconsistent': 0.0,
                'insufficient': 1.0,
                'contradictory': 0.0
            }
    
    def classify(self, backstory: str, evidence_chunks: List[Dict]) -> Dict:
        """
        Classify backstory causal consistency with structured output.
        
        Args:
            backstory: Backstory text
            evidence_chunks: Retrieved evidence chunks
            
        Returns:
            Dict with:
            - prediction: Binary output (0 or 1)
            - label: 4-class hypothesis label
            - scores: Hypothesis scores
            - evidence_sufficient: Boolean
            - reason: Explanation
        """
        # Check evidence sufficiency
        is_sufficient, sufficiency_reason = self._check_evidence_sufficiency(evidence_chunks)
        
        if not is_sufficient:
            return {
                'prediction': 0,
                'label': 'insufficient',
                'scores': {'consistent': 0.0, 'inconsistent': 0.0, 'insufficient': 1.0, 'contradictory': 0.0},
                'evidence_sufficient': False,
                'reason': sufficiency_reason
            }
        
        # Score hypotheses using LLM
        scores = self._score_hypotheses(backstory, evidence_chunks)
        
        # Python-based label selection: pick highest scoring hypothesis
        selected_label = max(scores.items(), key=lambda x: x[1])[0]
        
        # Map to binary prediction
        prediction = self.LABELS[selected_label]
        
        # Determine reason
        if selected_label == 'consistent':
            reason = f"Backstory is consistent (score: {scores['consistent']:.2f})"
        elif selected_label == 'contradictory':
            reason = f"Direct contradictions found (score: {scores['contradictory']:.2f})"
        elif selected_label == 'inconsistent':
            reason = f"Logical/causal inconsistencies (score: {scores['inconsistent']:.2f})"
        else:
            reason = f"Insufficient evidence to verify (score: {scores['insufficient']:.2f})"
        
        return {
            'prediction': prediction,
            'label': selected_label,
            'scores': scores,
            'evidence_sufficient': True,
            'reason': reason
        }
