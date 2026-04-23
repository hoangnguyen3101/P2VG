"""
Visualize CE (Clinical Entity) condition coverage in tokenizer vocabulary.
Shows how many vocab tokens match each CE condition keyword.
"""
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


CE_CONDITIONS = {
    "disc_herniation": ["disc herniation", "disc protrusion", "disc extrusion"],
    "disc_bulge": ["disc bulge", "bulging disc"],
    "disc_degeneration": ["degeneration", "degenerative", "signal reduction"],
    "annular_tear": ["annular tear", "annular fissure"],
    "disc_height_loss": ["disc height", "height loss", "height reduction"],
    "spinal_stenosis": ["spinal stenosis", "spinal canal stenosis", "canal stenosis"],
    "foraminal_stenosis": ["foraminal stenosis", "neural foraminal"],
    "lateral_recess_stenosis": ["lateral recess stenosis", "lateral recess narrowing"],
    "nerve_compression": ["nerve root compression", "nerve root contact", "nerve root impingement"],
    "spondylolisthesis": ["spondylolisthesis", "anterolisthesis", "retrolisthesis"],
    "compression_fracture": ["compression fracture", "vertebral fracture"],
    "osteophytes": ["osteophyte", "osteophytes"],
    "facet_arthropathy": ["facet joint", "facet hypertrophy", "facet arthropathy"],
    "curvature_abnormality": ["scoliosis", "kyphosis", "lordosis"],
}


def count_vocab_matches(tokenizer, keywords):
    """Count how many vocab tokens contain each keyword's individual words (case-insensitive)."""
    vocab = tokenizer.get_vocab()
    vocab_tokens_lower = [tok.lower().replace("▁", "").replace("##", "") for tok in vocab.keys()]
    
    matches = {}
    for kw in keywords:
        # Split multi-word keyword into individual words, match each
        words = kw.lower().split()
        total = 0
        for word in words:
            count = sum(1 for tok in vocab_tokens_lower if word in tok and len(tok) > 0)
            total += count
        matches[kw] = total
    return matches


def tokenize_analysis(tokenizer, keywords):
    """Show how each keyword gets tokenized."""
    results = {}
    for kw in keywords:
        token_ids = tokenizer.encode(kw, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        results[kw] = {
            "tokens": tokens,
            "num_tokens": len(tokens),
            "ids": token_ids,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./output_gemma3/merged_hf")
    parser.add_argument("--output_dir", type=str, default="./output_gemma3/tokenizer_overview")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Collect all unique keywords
    all_keywords = []
    for kws in CE_CONDITIONS.values():
        all_keywords.extend(kws)
    
    # ========================
    # 1. Tokenization breakdown per CE condition
    # ========================
    print("=" * 80)
    print("CE CONDITION TOKENIZATION ANALYSIS")
    print("=" * 80)
    
    cond_data = {}
    for cond_name, keywords in CE_CONDITIONS.items():
        print(f"\n{'─'*60}")
        print(f"  {cond_name.upper()}")
        print(f"{'─'*60}")
        avg_tokens = []
        for kw in keywords:
            tok_result = tokenize_analysis(tokenizer, [kw])
            r = tok_result[kw]
            avg_tokens.append(r["num_tokens"])
            tokens_str = " | ".join(r["tokens"])
            print(f"  \"{kw}\"")
            print(f"    → [{tokens_str}]  ({r['num_tokens']} tokens)")
        cond_data[cond_name] = np.mean(avg_tokens)
    
    # ========================
    # 2. Bar chart: avg tokens needed per condition
    # ========================
    fig, ax = plt.subplots(figsize=(14, 6))
    conditions = list(cond_data.keys())
    avg_counts = list(cond_data.values())
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(conditions)))
    
    bars = ax.barh(conditions, avg_counts, color=colors, edgecolor='white', height=0.7)
    for bar, val in zip(bars, avg_counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel("Average Number of Tokens per Keyword", fontsize=12)
    ax.set_title("CE Conditions: Tokenization Complexity\n(Fewer tokens = better vocabulary coverage)", 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(avg_counts) + 1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ce_tokenization.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {os.path.join(args.output_dir, 'ce_tokenization.png')}")
    
    # ========================
    # 3. Vocab match count per keyword
    # ========================
    fig, ax = plt.subplots(figsize=(16, 8))
    keyword_matches = {}
    for kw in all_keywords:
        matches = count_vocab_matches(tokenizer, [kw])
        keyword_matches[kw] = matches[kw]
    
    # Sort by count
    sorted_kw = sorted(keyword_matches.items(), key=lambda x: x[1], reverse=True)
    kw_names = [x[0] for x in sorted_kw]
    kw_counts = [x[1] for x in sorted_kw]
    
    colors = ['#4ECDC4' if c > 0 else '#FF6B6B' for c in kw_counts]
    bars = ax.barh(range(len(kw_names)), kw_counts, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(kw_names)))
    ax.set_yticklabels(kw_names, fontsize=9)
    
    for bar, val in zip(bars, kw_counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Number of Matching Tokens in Vocabulary", fontsize=12)
    ax.set_title("CE Keywords: Vocabulary Token Matches\n(Green = found, Red = not found as standalone token)", 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ce_vocab_matches.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(args.output_dir, 'ce_vocab_matches.png')}")
    
    # ========================
    # 4. Summary table
    # ========================
    print("\n" + "=" * 80)
    print(f"{'Keyword':<30} {'Vocab Matches':>15} {'Avg Tokens':>12}")
    print("-" * 60)
    for cond_name, keywords in CE_CONDITIONS.items():
        for kw in keywords:
            match_count = keyword_matches[kw]
            tok_result = tokenize_analysis(tokenizer, [kw])
            n_tok = tok_result[kw]["num_tokens"]
            print(f"  {kw:<28} {match_count:>13} {n_tok:>10}")
        print()


if __name__ == "__main__":
    main()
