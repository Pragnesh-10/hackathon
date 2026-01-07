"""Hardened claim-based classifier with strict Python rules and validation."""
import os
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import json
import time

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Global evidence log
EVIDENCE_LOG = []


def get_claim_specific_evidence(claim_text, all_chunks, top_k=3):
    """Get top K most relevant chunks for specific claim (not global)."""
    claim_words = set(claim_text.lower().split())
    
    scored_chunks = []
    for chunk in all_chunks:
        chunk_words = set(chunk['text'].lower().split())
        overlap = len(claim_words & chunk_words)
        score = overlap / (len(claim_words) + 1)
        scored_chunks.append({**chunk, 'score': score})
    
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:top_k]


def classify_backstory_hardened(backstory, book_chunks, story_id="unknown"):
    """
    Hardened claim-based classification with strict Python rules.
    
    All prediction logic is in Python, not LLM.
    Fail-closed: errors and invalid outputs → prediction = 0.
    """
    try:
        # Step 1: Decompose backstory into claims with key_claim flag
        decompose_prompt = f"""Break this backstory into atomic claims. Mark critical claims.

Backstory: {backstory}

Categorize claims:
- childhood: Youth events
- belief: Core beliefs, values
- affiliation: Groups, relationships  
- irreversible_action: Death, imprisonment, loss of ability, public commitment

Mark key_claim=true for:
- irreversible_action (always)
- affiliation (always)
- belief (if core/defining)

Output ONLY valid JSON:
{{
  "claims": [
    {{
      "claim_id": 0,
      "claim": "text",
      "claim_type": "childhood|belief|affiliation|irreversible_action",
      "key_claim": true/false
    }}
  ]
}}"""

        print(f"      📋 Decomposing claims...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Output only valid JSON."},
                {"role": "user", "content": decompose_prompt}
            ],
            temperature=0.0,
            max_tokens=600
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Clean JSON
        if '```' in answer:
            answer = answer.split('```')[1]
            if answer.startswith('json'):
                answer = answer[4:]
        
        decompose_result = json.loads(answer.strip())
        
        # STRICT VALIDATION: Required fields
        if 'claims' not in decompose_result:
            print(f"      ❌ Invalid JSON: missing 'claims' field")
            return fail_closed(story_id, "Invalid JSON structure")
        
        claims = decompose_result['claims']
        
        # Validate each claim has required fields
        for claim in claims:
            required_fields = ['claim_id', 'claim', 'claim_type', 'key_claim']
            for field in required_fields:
                if field not in claim:
                    print(f"      ❌ Claim missing field: {field}")
                    return fail_closed(story_id, f"Claim missing {field}")
        
        print(f"      ✅ Extracted {len(claims)} claims")
        
        # Step 2: For each claim, get specific evidence and classify
        claim_results = []
        has_contradiction = False
        
        for claim_obj in claims:
            claim_id = claim_obj['claim_id']
            claim_text = claim_obj['claim']
            claim_type = claim_obj['claim_type']
            key_claim = claim_obj.get('key_claim', False)
            
            print(f"      🔍 Claim {claim_id}: [{claim_type}] {'KEY' if key_claim else 'non-key'}")
            
            # Get top 2-3 chunks SPECIFIC to this claim
            claim_evidence = get_claim_specific_evidence(claim_text, book_chunks, top_k=3)
            
            # Build evidence text
            evidence_text = "\n\n---\n\n".join([
                f"[Chunk {chunk['chunk_index']}]\n{chunk['text'][:800]}"
                for chunk in claim_evidence
            ])
            
            # Classify evidence for this claim
            classify_prompt = f"""Classify how evidence relates to claim.

Claim: {claim_text}

Evidence:
{evidence_text}

Rules:
- "supported": Evidence confirms claim
- "contradicted": Direct factual conflict OR timeline violation
- "insufficient": No clear evidence
- "mixed": Both supporting and contradicting evidence found

Timeline violations:
- Event occurs AFTER death/imprisonment/irreversible action

Output ONLY valid JSON:
{{
  "evidence_status": "supported|contradicted|insufficient|mixed",
  "rationale": "brief explanation"
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": classify_prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Clean JSON
            if '```' in answer:
                answer = answer.split('```')[1]
                if answer.startswith('json'):
                    answer = answer[4:]
            
            classify_result = json.loads(answer.strip())
            
            # STRICT VALIDATION
            if 'evidence_status' not in classify_result:
                print(f"         ❌ Missing evidence_status")
                return fail_closed(story_id, "Missing evidence_status in claim classification")
            
            evidence_status = classify_result['evidence_status']
            rationale = classify_result.get('rationale', 'No rationale provided')
            
            # PYTHON TIMELINE ENFORCEMENT
            # If irreversible action and rationale mentions "after", force contradiction
            if claim_type == 'irreversible_action':
                if 'after' in rationale.lower() or 'subsequent' in rationale.lower():
                    print(f"         🚨 TIMELINE VIOLATION detected in rationale")
                    evidence_status = 'contradicted'
                    rationale += " [Timeline violation enforced by Python]"
            
            # Log evidence
            for chunk in claim_evidence[:2]:  # Log top 2 chunks
                EVIDENCE_LOG.append({
                    'story_id': story_id,
                    'claim_id': claim_id,
                    'chunk_id': chunk['chunk_index'],
                    'evidence_status': evidence_status,
                    'rationale': rationale[:200]
                })
            
            # Check for failures
            if evidence_status == 'contradicted':
                has_contradiction = True
                print(f"         ❌ CONTRADICTED: {rationale}")
            elif evidence_status == 'mixed':
                has_contradiction = True  # Mixed = fail (conservative)
                print(f"         ⚠️  MIXED: {rationale}")
            elif evidence_status == 'insufficient' and key_claim:
                has_contradiction = True  # Key claim without evidence = fail
                print(f"         ⚠️  INSUFFICIENT (key claim): {rationale}")
            else:
                emoji = "✅" if evidence_status == 'supported' else "⚠️"
                print(f"         {emoji} {evidence_status.upper()}: {rationale}")
            
            claim_results.append({
                'claim_id': claim_id,
                'claim': claim_text,
                'claim_type': claim_type,
                'key_claim': key_claim,
                'evidence_status': evidence_status,
                'rationale': rationale
            })
        
        # PYTHON-ONLY PREDICTION LOGIC (NOT MODEL CONTROLLED)
        if has_contradiction:
            prediction = 0
            label = 'inconsistent'
        else:
            # Check if any key claim is insufficient
            key_insufficient = any(
                c['key_claim'] and c['evidence_status'] == 'insufficient'
                for c in claim_results
            )
            if key_insufficient:
                prediction = 0
                label = 'inconsistent'
            else:
                prediction = 1
                label = 'consistent'
        
        # FIXED CONFIDENCE CALCULATION (only subtract, never add)
        confidence = 1.0
        for claim in claim_results:
            status = claim['evidence_status']
            if status == 'contradicted':
                confidence -= 0.7
            elif status in ['mixed', 'insufficient']:
                confidence -= 0.3
            # supported → no change (0)
        
        confidence = max(0.05, confidence)  # Floor at 0.05
        
        reason = f"{len([c for c in claim_results if c['evidence_status'] == 'contradicted'])} contradictions, " \
                 f"{len([c for c in claim_results if c['evidence_status'] in ['mixed', 'insufficient']])} weak conflicts"
        
        return {
            'prediction': prediction,
            'label': label,
            'confidence': confidence,
            'claims': claim_results,
            'has_hard_contradiction': has_contradiction,
            'reason': reason
        }
        
    except json.JSONDecodeError as e:
        print(f"      ❌ JSON Parse Error: {e}")
        return fail_closed(story_id, f"JSON parse failed: {e}")
    
    except Exception as e:
        print(f"      ❌ Unexpected Error: {e}")
        return fail_closed(story_id, f"Unexpected error: {e}")


def fail_closed(story_id, reason):
    """Hard fail: Always return prediction=0 on errors."""
    EVIDENCE_LOG.append({
        'story_id': story_id,
        'claim_id': -1,
        'chunk_id': -1,
        'evidence_status': 'error',
        'rationale': reason
    })
    
    return {
        'prediction': 0,  # NEVER default to 1
        'label': 'insufficient',
        'confidence': 0.05,
        'claims': [],
        'has_hard_contradiction': True,
        'reason': reason
    }


def main():
    """Run hardened classifier."""
    global EVIDENCE_LOG
    EVIDENCE_LOG = []
    
    print("="*70)
    print("🛡️  HARDENED CLAIM-BASED CLASSIFIER")
    print("="*70)
    print("Python-only prediction logic | Fail-closed validation\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("❌ ERROR: OPENAI_API_KEY not set!")
        return
    
    print(f"✅ API Key loaded\n")
    
    # Load data
    print("📂 Loading dataset...")
    train_file = "data/Dataset/train.csv"
    books_dir = "data/Dataset/Books"
    
    train_df = pd.read_csv(train_file)
    
    books = {}
    for book_file in Path(books_dir).glob("*.txt"):
        with open(book_file, 'r', encoding='utf-8') as f:
            books[book_file.stem] = f.read()
    
    print(f"   Train: {len(train_df)} examples")
    print(f"   Books: {len(books)} loaded\n")
    
    os.makedirs("output", exist_ok=True)
    
    print("="*70)
    print("🧪 Testing on Faria, Noirtier, and Sample")
    print("="*70)
    
    results = []
    
    # Test: Faria (137), Noirtier (109, 104), Thalcave (46), Kai-Koumou (74)
    test_ids = [137, 109, 104, 46, 74]
    
    for idx, example_id in enumerate(test_ids):
        row = train_df[train_df['id'] == example_id].iloc[0]
        
        book_name = row['book_name']
        character = row['char']
        backstory = row['content']
        true_label = row.get('label', '?')
        
        print(f"\n{'─'*70}")
        print(f"🔍 [{idx+1}/{len(test_ids)}] ID={example_id}: {character}")
        print(f"   📖 Book: {book_name}")
        print(f"   ✓ True: {true_label}")
        print(f"   📝 Backstory: {backstory[:80]}...")
        
        # Find book
        matched_book = None
        for book_key, book_text in books.items():
            if book_key.lower() in book_name.lower():
                matched_book = book_text
                break
        
        if not matched_book:
            matched_book = list(books.values())[0]
        
        # Chunk book
        chunks = []
        for i in range(0, len(matched_book), 5000):
            chunks.append({
                'text': matched_book[i:i+5000],
                'chunk_index': len(chunks)
            })
        
        print(f"   📦 {len(chunks)} chunks available")
        
        # Classify with hardened approach
        result = classify_backstory_hardened(backstory, chunks, story_id=str(example_id))
        
        print(f"   🎯 PREDICTION: {result['prediction']} ({result['label']}, conf={result['confidence']:.2f})")
        print(f"   💭 {result['reason']}")
        
        results.append({
            'id': example_id,
            'character': character,
            'prediction': result['prediction'],
            'label': result['label'],
            'confidence': result['confidence'],
            'true_label': true_label,
            'num_claims': len(result['claims']),
            'reason': result['reason']
        })
        
        # Rate limit delay
        if idx < len(test_ids) - 1:
            print(f"   ⏱️  Waiting 5s...")
            time.sleep(5)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = "output/results.csv"
    results_df.to_csv(output_file, index=False)
    
    # Save evidence log
    evidence_file = "output/evidence_log.json"
    with open(evidence_file, 'w') as f:
        json.dump(EVIDENCE_LOG, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ Hardened Classification Complete!")
    print('='*70)
    print(f"Total: {len(results_df)}")
    print(f"Predictions: {list(results_df['prediction'])}")
    print(f"True labels: {list(results_df['true_label'])}")
    
    # Accuracy
    label_map = {'consistent': 1, 'contradict': 0, 'inconsistent': 0}
    true_binary = results_df['true_label'].map(label_map)
    accuracy = (results_df['prediction'] == true_binary).mean()
    print(f"Accuracy: {accuracy:.2%}")
    
    print(f"\n📊 Summary:")
    print(f"   Consistent: {sum(results_df['prediction'] == 1)}/5")
    print(f"   Inconsistent: {sum(results_df['prediction'] == 0)}/5")
    print(f"   Avg confidence: {results_df['confidence'].mean():.2f}")
    
    print(f"\n📄 Results: {output_file}")
    print(f"📄 Evidence log: {evidence_file}")


if __name__ == "__main__":
    main()
