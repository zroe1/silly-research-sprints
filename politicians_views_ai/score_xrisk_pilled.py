#!/usr/bin/env python3
"""
Score how XRisk-pilled each politician is (0-100) using GPT-4o logit aggregation.
Single token generation with probability distribution over 0-100.
"""

import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path

# Setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
SCRIPT_DIR = Path(__file__).parent
PROGRESS_FILE = SCRIPT_DIR / "progress.json"
OUTPUT_FILE = SCRIPT_DIR / "xrisk_pilled_scores.json"

def load_progress():
    """Load progress from progress.json if it exists."""
    if not PROGRESS_FILE.exists():
        return {}
    with open(PROGRESS_FILE, 'r') as f:
        return json.load(f)

def get_politician_info(name, progress_data):
    """Get raw JSON data for a politician from progress.json."""
    if name not in progress_data.get('results', {}):
        return None
    return progress_data['results'][name]

def score_xrisk_pilled(politician_info):
    """
    Score how XRisk-pilled someone is using logit aggregation.
    Returns a 0-100 score based on probability distribution.
    """
    
    prompt = f"""Based on the following information about a politician's views on AI, rate how "XRisk-pilled" they are on a scale of 0-100.

Signs of being XRisk-pilled include:
- Believe AI could cause existential risk
- Thinks that alignment of AGI is a serious concern
- Believes that AI safety should be taken seriously by congress

NOTE: We are measuring how much the politician cares about XRisk, not how much they are concerned about US-China relations.
A politician who is most XRisk-pilled is someone who thinks AI is a risk to humanity -- not just the US.

Scale:
0 = Does not mention XRisk in their views on AI or does not think XRisk is a serious concern
50 = Cites XRisk when talking about AI but doesn't take it seriously
100 = Cites XRisk as being a serious concern that is imperative for congress to address

{json.dumps(politician_info, indent=2)}

Rate them from 0-100 (respond with ONLY a number):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20
        )
        
        # Get the logprobs for the top tokens
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        
        # Extract probabilities for 0-100
        score_probs = {}
        for token_info in logprobs:
            token = token_info.token.strip()
            if token.isdigit():
                num = int(token)
                if 0 <= num <= 100:
                    # Convert log probability to probability
                    prob = np.exp(token_info.logprob)
                    score_probs[num] = prob
        
        if not score_probs:
            # Fallback: use the actual generated token
            generated = response.choices[0].message.content.strip()
            if generated.isdigit():
                return int(generated), {}
            return None, {}
        
        # Calculate expected value (weighted average)
        total_prob = sum(score_probs.values())
        if total_prob == 0:
            return None, {}
            
        # Normalize and calculate expected value
        expected_score = sum(score * (prob/total_prob) 
                           for score, prob in score_probs.items())
        
        # Also return the distribution for analysis
        normalized_probs = {score: prob/total_prob 
                           for score, prob in score_probs.items()}
        
        return round(expected_score, 1), normalized_probs
        
    except Exception as e:
        print(f"  Error: {e}")
        return None, {}

def main():
    print("="*80)
    print("XRisk-Pilled Scoring Script")
    print("="*80)
    
    # Load progress data
    print("\nLoading politician AI views from progress.json...")
    progress_data = load_progress()
    
    if not progress_data or 'results' not in progress_data:
        print("ERROR: No progress.json found. Run analyze_politicians_ai_views.py first.")
        return
    
    politicians = progress_data['results']
    print(f"Found {len(politicians)} politicians with AI analysis")
    
    # Score each politician
    results = {}
    
    for idx, name in enumerate(politicians.keys(), 1):
        print(f"\n[{idx}/{len(politicians)}] Scoring {name}...")
        
        # Get politician info
        info = get_politician_info(name, progress_data)
        if not info:
            print("  No info available")
            continue
        if info['analysis']['overall_stance'] == 'unknown':
            print("  Overall stance is unknown")
            continue
        
        # Score them
        score, distribution = score_xrisk_pilled(info)
        
        if score is not None:
            print(f"  Score: {score}/100")
            
            # Get their party and state
            pol_data = politicians[name]
            results[name] = {
                "score": score,
                "party": pol_data.get('party'),
                "state": pol_data.get('state'),
                "type": pol_data.get('type'),
                "distribution": distribution
            }
        else:
            print("  Failed to score")
    
    # Save results
    print(f"\n{'='*80}")
    print(f"Scoring complete! Scored {len(results)}/{len(politicians)} politicians")
    print(f"Saving to {OUTPUT_FILE}")
    
    # Add some statistics
    scores = [r['score'] for r in results.values()]
    output = {
        "summary": {
            "total_scored": len(results),
            "mean_score": round(np.mean(scores), 1),
            "median_score": round(np.median(scores), 1),
            "std_dev": round(np.std(scores), 1),
            "min_score": min(scores),
            "max_score": max(scores)
        },
        "scores": results
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSummary Statistics:")
    print(f"  Mean: {output['summary']['mean_score']}")
    print(f"  Median: {output['summary']['median_score']}")
    print(f"  Range: {output['summary']['min_score']}-{output['summary']['max_score']}")
    
    # Top 10 most XRisk-pilled
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    print(f"\nTop 10 Most XRisk-Pilled:")
    for i, (name, data) in enumerate(sorted_results[:10], 1):
        print(f"  {i}. {name} ({data['party']}-{data['state']}): {data['score']}")

if __name__ == "__main__":
    main()

