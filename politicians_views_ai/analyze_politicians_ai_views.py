#!/usr/bin/env python3
"""
Script to analyze politicians' views on AI using Claude 4.5 Sonnet with web search.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, List, Optional
import anthropic
from pathlib import Path

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Please set ANTHROPIC_API_KEY environment variable")

MODEL = "claude-sonnet-4-5-20250929"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# File paths
SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "legislators-current.csv"
OUTPUT_PATH = SCRIPT_DIR / "politicians_ai_views.json"
PROGRESS_PATH = SCRIPT_DIR / "progress.json"

# Rate limiting
DELAY_BETWEEN_CALLS = 2  # seconds


def load_progress() -> Dict:
    """Load progress from previous run if exists."""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, 'r') as f:
            return json.load(f)
    return {"completed": [], "results": {}}


def save_progress(progress: Dict):
    """Save progress to file."""
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2)


def call_claude_with_search(prompt: str, max_retries: int = 3) -> Optional[Dict]:
    """Call Claude API with web search enabled."""
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8000,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 10  # Allow up to 10 searches per politician
                }]
            )
            
            # Extract text content and search usage
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
            
            # Get usage stats
            usage_info = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "web_searches": 0
            }
            
            if hasattr(response.usage, 'server_tool_use'):
                usage_info['web_searches'] = getattr(
                    response.usage.server_tool_use, 
                    'web_search_requests', 
                    0
                )
            
            return {
                "content": text_content,
                "usage": usage_info
            }
            
        except anthropic.RateLimitError as e:
            print(f"  Rate limit hit: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  All retries exhausted")
                return None
                
        except anthropic.APIError as e:
            print(f"  API Error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  All retries exhausted")
                return None
                
        except Exception as e:
            print(f"  Unexpected error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  All retries exhausted")
                return None
    
    return None


def create_prompt(politician_name: str) -> str:
    """Create the analysis prompt for a politician."""
    return f"""I am interested in the following elected official: {politician_name}.

I would like you to find 10 quotes to summarize their attitude about AI. I am especially interested in the following:

- Do they reference China or other foreign countries when speaking about AI?
- Do they appear "AGI-pilled"?
- Do they talk about AI as being transformative or overhyped?
- Do they ever talk about existential risk from AI?
- Do they speak about AI safety or alignment?
- Have they made public statements about AI companies or key figures in the field?

Do a deep research and come back with your top 10 quotes to summarize their feelings about AI.

Please format your response as a JSON object with the following structure:
{{
    "politician_name": "{politician_name}",
    "summary": "Brief 2-3 sentence summary of their overall AI stance",
    "quotes": [
        {{
            "quote": "The actual quote text",
            "source": "Source/date if available",
            "context": "Brief context about the quote",
        }}
    ],
    "analysis": {{
        "mentions_china": true/false,
        "mentions_foreign_countries": true/false,
        "agi_pilled": true/false,
        "views_as_transformative": true/false,
        "views_as_overhyped": true/false,
        "discusses_existential_risk": true/false,
        "discusses_safety_alignment": true/false,
        "mentions_ai_companies": true/false,
        "overall_stance": "supportive/cautious/critical/neutral/unknown"
    }},
    "notes": "Any additional relevant information"
}}

If you cannot find sufficient information about their AI views, return a response indicating limited information is available."""


def parse_politician_response(response: str) -> Dict:
    """Parse the API response, extracting JSON if embedded in text."""
    # Try to extract JSON from markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        json_str = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        json_str = response[start:end].strip()
    else:
        json_str = response.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If parsing fails, return raw response
        return {
            "raw_response": response,
            "parse_error": "Could not parse as JSON"
        }


def main():
    """Main execution function."""
    print("=" * 80)
    print("Politicians AI Views Analysis Script")
    print("=" * 80)
    
    # Load CSV
    print(f"\nLoading politicians from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} politicians")
    
    # Load progress
    progress = load_progress()
    completed = set(progress.get("completed", []))
    results = progress.get("results", {})
    
    if completed:
        print(f"Resuming from previous run. {len(completed)} already completed.")
    
    # Process each politician
    total = len(df)
    for idx, row in df.iterrows():
        # Create full name
        full_name = row['full_name']
        
        # Skip if already completed
        if full_name in completed:
            print(f"\n[{idx+1}/{total}] Skipping {full_name} (already completed)")
            continue
        
        print(f"\n[{idx+1}/{total}] Analyzing {full_name}")
        print(f"  Position: {row['type']} from {row['state']}")
        
        # Create and send prompt
        prompt = create_prompt(full_name)
        response = call_claude_with_search(prompt)
        
        if response:
            parsed_result = parse_politician_response(response['content'])
            parsed_result['full_name'] = full_name
            parsed_result['state'] = row['state']
            parsed_result['type'] = row['type']
            parsed_result['party'] = row['party']
            parsed_result['usage'] = response['usage']
            
            results[full_name] = parsed_result
            completed.add(full_name)
            
            # Display usage stats
            usage = response['usage']
            print(f"  ✓ Analysis complete")
            print(f"    Web searches: {usage['web_searches']}")
            print(f"    Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out")
            
            # Save progress after each successful completion
            progress = {
                "completed": list(completed),
                "results": results
            }
            save_progress(progress)
        else:
            print(f"  ✗ Failed to get response")
        
        # Rate limiting
        if idx < total - 1:  # Don't delay after last item
            time.sleep(DELAY_BETWEEN_CALLS)
    
    # Save final results
    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"Processed {len(completed)} politicians")
    print(f"Saving results to: {OUTPUT_PATH}")
    
    final_output = {
        "metadata": {
            "total_politicians": total,
            "analyzed": len(completed),
            "model": MODEL,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": results
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"✓ Results saved successfully")
    print(f"\nYou can delete {PROGRESS_PATH} if you want to start fresh next time.")


if __name__ == "__main__":
    main()

