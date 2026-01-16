#!/usr/bin/env python3
"""
Plot politicians' ideology scores vs AGI-pilled scores.
Single file, no dependencies beyond standard plotting libs.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm
from adjustText import adjust_text

# Add the font file
font_path = 'Montserrat-VariableFont_wght.ttf'
fm.fontManager.addfont(font_path)
bold_prop = fm.FontProperties(fname="Montserrat-Bold.ttf", size=20)
semibold_prop = fm.FontProperties(fname="Montserrat-SemiBold.ttf")

# Set as default
plt.rcParams['font.family'] = 'Montserrat'

# Setup
SCRIPT_DIR = Path(__file__).parent
LEGISLATORS_CSV = SCRIPT_DIR / "legislators-current.csv"
HOUSE_FILE = SCRIPT_DIR / "sponsorshipanalysis_h.txt"
SENATE_FILE = SCRIPT_DIR / "sponsorshipanalysis_s.txt"
AGI_SCORES_FILE = SCRIPT_DIR / "agi_pilled_scores.json"
OUTPUT_FILE = SCRIPT_DIR / "ideology_vs_agi_plot.png"

# Brand colors from CSS
COLORS = {
    'maroon': '#800000',
    'maroon_light': '#9A0000',
    'maroon_dark': '#5E0000',
    'yellow': '#FFD100',
    'orange': '#D45D00',
    'gray_dark': '#3D3D3D',
    'gray_medium': '#767676',
    'text_color': '#1a1a1a',
    'blue': '#0000FF',
    'white': '#FFFFFF'
}

def main():
    print("Loading data...")
    
    # Load legislators to get govtrack_id -> full_name mapping
    legislators = pd.read_csv(LEGISLATORS_CSV)
    govtrack_to_name = dict(zip(legislators['govtrack_id'], legislators['full_name']))
    
    # Load ideology scores from sponsorship analysis
    house_df = pd.read_csv(HOUSE_FILE)
    senate_df = pd.read_csv(SENATE_FILE)
    ideology_df = pd.concat([house_df, senate_df])
    
    # Create govtrack_id -> ideology mapping
    ideology_map = dict(zip(ideology_df['ID'], ideology_df['ideology']))
    
    # Load AGI-pilled scores
    with open(AGI_SCORES_FILE, 'r') as f:
        agi_data = json.load(f)
    
    # Match up the data
    data = []
    for name, score_data in agi_data['scores'].items():
        # Find govtrack_id for this politician
        govtrack_id = None
        for gid, leg_name in govtrack_to_name.items():
            if leg_name == name:
                govtrack_id = gid
                break
        
        if govtrack_id is None:
            continue
            
        # Get ideology score
        ideology = ideology_map.get(govtrack_id)
        if ideology is None:
            continue
        
        # Add to dataset
        data.append({
            'name': name,
            'ideology': ideology,
            'agi_score': score_data['score'],
            'party': score_data['party']
        })
    
    print(f"Matched {len(data)} politicians with both scores")
    
    # Create plot with brand styling
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['white'])
    ax.set_facecolor(COLORS['white'])
    
    # Separate by party
    democrats = [d for d in data if d['party'] == 'Democrat']
    republicans = [d for d in data if d['party'] == 'Republican']
    independents = [d for d in data if d['party'] == 'Independent']
    
    # Plot with brand colors
    if democrats:
        ax.scatter(
            [d['ideology'] for d in democrats],
            [d['agi_score'] for d in democrats],
            c=COLORS['blue'], alpha=1, s=50, label='Democrat',
            # edgecolors=COLORS['gray_dark'], linewidths=0
        )
    
    if republicans:
        ax.scatter(
            [d['ideology'] for d in republicans],
            [d['agi_score'] for d in republicans],
            c=COLORS['maroon'], alpha=1, s=50, label='Republican',
            # edgecolors=COLORS['gray_dark'], linewidths=0.5
        )
    
    if independents:
        ax.scatter(
            [d['ideology'] for d in independents],
            [d['agi_score'] for d in independents],
            c=COLORS['yellow'], alpha=1, s=50, label='Independent',
            # edgecolors=COLORS['gray_dark'], linewidths=0.5
        )
    
    # Labels and formatting with brand fonts and colors
    ax.set_xlabel('Ideology Score (0=Progressive, 1=Conservative)', 
                  fontsize=13, color=COLORS['text_color'], 
                  fontfamily='sans-serif', fontweight=500, fontproperties=semibold_prop)
    ax.set_ylabel('AGI-Pilled Score (0-100)', 
                  fontsize=13, color=COLORS['text_color'],
                  fontfamily='sans-serif', fontweight=500, fontproperties=semibold_prop)
    ax.set_title('Political Ideology vs AGI-Pilled Score', 
                 fontsize=30, fontweight=600, color=COLORS['text_color'],
                pad=20, fontproperties=bold_prop)
    
    # Style legend
    legend = ax.legend(frameon=True, fancybox=False, shadow=False,
                      fontsize=11, loc='center left') # middle left
    legend.get_frame().set_facecolor(COLORS['white'])
    legend.get_frame().set_edgecolor(COLORS['gray_medium'])
    legend.get_frame().set_linewidth(1)
    
    # Grid styling
    ax.grid(True, alpha=0.25, color=COLORS['gray_medium'], linewidth=0.5)
    
    # Tick styling
    ax.tick_params(colors=COLORS['text_color'], labelsize=10)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['gray_medium'])
        spine.set_linewidth(1)
    
    # # Add some statistics
    mean_ideology = sum(d['ideology'] for d in data) / len(data)
    mean_agi = sum(d['agi_score'] for d in data) / len(data)
    
    # # Add reference lines with brand colors
    # ax.axhline(y=mean_agi, color=COLORS['gray_medium'], linestyle='--', 
    #            alpha=0.4, linewidth=1.5, zorder=0)
    # ax.axvline(x=mean_ideology, color=COLORS['gray_medium'], linestyle='--', 
    #            alpha=0.4, linewidth=1.5, zorder=0)
    
    # Highlight extreme points with collision avoidance
    top_agi = sorted(data, key=lambda x: x['agi_score'], reverse=True)[:21]

    # Normalize coordinates for distance calculation (since x and y have different scales)
    x_range = max(d['ideology'] for d in data) - min(d['ideology'] for d in data)
    y_range = max(d['agi_score'] for d in data) - min(d['agi_score'] for d in data)

    labeled_points = []  # Track points that have labels
    min_distance = 0.08  # Adjust this threshold as needed

    for person in top_agi:
        # Normalize coordinates
        norm_x = person['ideology'] / x_range
        norm_y = person['agi_score'] / y_range
        
        # Check distance to all already-labeled points
        too_close = False
        for (lx, ly) in labeled_points:
            dist = ((norm_x - lx)**2 + (norm_y - ly)**2)**0.5
            if dist < min_distance:
                too_close = True
                break
        
        if too_close:
            continue  # Skip this label
        
        # Add label
        labeled_points.append((norm_x, norm_y))
        
        nameparts = person['name'].split()
        if nameparts[-1] == 'Jr.':
            name = nameparts[-2] + ' ' + nameparts[-1]
        else:
            name = nameparts[-1]
        
        ax.annotate(
            name,
            (person['ideology'], person['agi_score']),
            xytext=(0, 8),
            textcoords='offset points',
            fontsize=8,
            color=COLORS['text_color'],
            fontweight=500,
            alpha=0.8,
            ha='center',
            va='bottom'
        )

    # Automatically adjust positions to avoid overlaps
    # adjusted_texts = adjust_text(texts)
    # # Highlight extreme points
    # top_agi = sorted(data, key=lambda x: x['agi_score'], reverse=True)[:21]
    # for person in top_agi:
    #     nameparts = person['name'].split()
    #     print(nameparts)
    #     if nameparts[-1] == 'Jr.':
    #         name = nameparts[-2] + ' ' + nameparts[-1]
    #     else:
    #         name = nameparts[-1]
    #     ax.annotate(
    #         name,  # Last name only
    #         (person['ideology'], person['agi_score']),
    #         xytext=(0, 8),  # Positive y = above the point
    #         textcoords='offset points',
    #         fontsize=8,
    #         color=COLORS['text_color'],
    #         fontweight=500,
    #         alpha=0.8,
    #         ha='center',  # Center horizontally
    #         va='bottom'   # Anchor from bottom of text
    #     )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', 
                facecolor=COLORS['white'], edgecolor='none')
    print(f"\n✓ Plot saved to {OUTPUT_FILE}")
    
    # Print some interesting statistics
    print(f"\nStatistics:")
    print(f"  Total politicians: {len(data)}")
    print(f"  Mean ideology: {mean_ideology:.3f}")
    print(f"  Mean AGI score: {mean_agi:.1f}")
    
    print(f"\nTop 5 Most AGI-Pilled:")
    for i, person in enumerate(top_agi[:5], 1):
        print(f"  {i}. {person['name']} ({person['party']}) - AGI: {person['agi_score']:.1f}, Ideology: {person['ideology']:.2f}")
    
    # Correlation
    import numpy as np
    ideologies = [d['ideology'] for d in data]
    agi_scores = [d['agi_score'] for d in data]
    correlation = np.corrcoef(ideologies, agi_scores)[0, 1]
    print(f"\nCorrelation (ideology vs AGI-pilled): {correlation:.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()

