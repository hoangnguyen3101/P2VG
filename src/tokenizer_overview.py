"""
Tokenizer & Vocabulary Overview for MedGemma-based P2VG pipeline.
Generates statistics and visualizations for demonstration.
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import Counter
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenizer vocabulary overview")
    parser.add_argument("--model_path", type=str, default="./output_gemma3/merged_hf",
                        help="Path to model/tokenizer directory")
    parser.add_argument("--output_dir", type=str, default="./output_gemma3/tokenizer_overview",
                        help="Directory to save visualizations")
    return parser.parse_args()


def analyze_vocab(tokenizer):
    """Analyze vocabulary composition."""
    vocab = tokenizer.get_vocab()
    total = len(vocab)
    
    categories = {
        "Special Tokens": [],
        "Medical/Clinical": [],
        "Subword (##/▁)": [],
        "Punctuation/Symbol": [],
        "Number": [],
        "CJK/Unicode": [],
        "Regular Words": [],
    }

    medical_keywords = [
        "disc", "spine", "spinal", "lumbar", "vertebr", "stenosis", "hernia",
        "nerve", "canal", "foramen", "radicul", "sclerosis", "osteo", "spondyl",
        "fracture", "compress", "bulge", "degenerat", "annular", "facet",
        "ligament", "cord", "marrow", "edema", "lesion", "tumor", "cyst",
        "cartilage", "joint", "bone", "mri", "ct", "imaging", "sagittal",
        "axial", "coronal", "anterior", "posterior", "lateral", "medial",
        "cervical", "thoracic", "sacral", "dorsal", "ventral",
        "diagnosis", "patholog", "symptom", "clinical", "patient",
        "mild", "moderate", "severe", "chronic", "acute",
    ]

    for token, idx in vocab.items():
        token_lower = token.lower().replace("▁", "").replace("##", "")
        
        if token.startswith("<") or token.startswith("[") or idx < 10:
            categories["Special Tokens"].append(token)
        elif any(kw in token_lower for kw in medical_keywords):
            categories["Medical/Clinical"].append(token)
        elif token.startswith("▁") or token.startswith("##"):
            if token_lower.isdigit():
                categories["Number"].append(token)
            else:
                categories["Subword (##/▁)"].append(token)
        elif all(not c.isalnum() for c in token):
            categories["Punctuation/Symbol"].append(token)
        elif token_lower.isdigit():
            categories["Number"].append(token)
        elif any(ord(c) > 127 for c in token):
            categories["CJK/Unicode"].append(token)
        else:
            categories["Regular Words"].append(token)

    return categories, total


def plot_vocab_composition(categories, total, output_dir):
    """Pie chart of vocabulary composition."""
    labels = []
    sizes = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    for cat, tokens in categories.items():
        if len(tokens) > 0:
            labels.append(f"{cat}\n({len(tokens):,})")
            sizes.append(len(tokens))

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors[:len(sizes)],
        autopct='%1.1f%%', startangle=90, pctdistance=0.85,
        textprops={'fontsize': 9}
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    ax.set_title(f"MedGemma Tokenizer Vocabulary Composition\n(Total: {total:,} tokens)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vocab_composition.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_medical_tokens(categories, output_dir):
    """Bar chart of medical token subcategories."""
    medical_tokens = categories.get("Medical/Clinical", [])
    if not medical_tokens:
        return
    
    subcategories = {
        "Anatomy": ["spine", "spinal", "lumbar", "vertebr", "cervical", "thoracic", "sacral", 
                     "disc", "cord", "nerve", "bone", "joint", "cartilage", "ligament", "marrow"],
        "Pathology": ["stenosis", "hernia", "fracture", "degenerat", "spondyl", "osteo", 
                      "bulge", "compress", "annular", "sclerosis", "edema", "lesion", "tumor", "cyst"],
        "Position": ["anterior", "posterior", "lateral", "medial", "dorsal", "ventral",
                     "sagittal", "axial", "coronal", "foramen", "canal"],
        "Severity": ["mild", "moderate", "severe", "chronic", "acute"],
        "Clinical": ["diagnosis", "patholog", "symptom", "clinical", "patient", "mri", "ct", "imaging"],
    }
    
    sub_counts = {}
    for sub_name, keywords in subcategories.items():
        count = 0
        for token in medical_tokens:
            token_lower = token.lower().replace("▁", "").replace("##", "")
            if any(kw in token_lower for kw in keywords):
                count += 1
        sub_counts[sub_name] = count

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(sub_counts.keys(), sub_counts.values(), 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD'])
    
    for bar, count in zip(bars, sub_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Medical Token Distribution by Subcategory", fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of Tokens")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "medical_tokens.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_special_tokens(tokenizer, output_dir):
    """Visualize special tokens and their IDs."""
    special_info = {
        "<pad>": tokenizer.pad_token_id,
        "<eos>": tokenizer.eos_token_id,
        "<bos>": tokenizer.bos_token_id,
        "<im_patch>": tokenizer.convert_tokens_to_ids("<im_patch>"),
        "<bx_start>": tokenizer.convert_tokens_to_ids("<bx_start>"),
        "<bx_end>": tokenizer.convert_tokens_to_ids("<bx_end>"),
    }
    
    # Add Gemma3 multimodal tokens if they exist
    for name in ["boi_token", "eoi_token"]:
        if name in tokenizer.special_tokens_map:
            tok = tokenizer.special_tokens_map[name]
            special_info[tok] = tokenizer.convert_tokens_to_ids(tok)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    table_data = [["Token", "Token ID", "Purpose"]]
    purposes = {
        "<pad>": "Padding",
        "<eos>": "End of sequence",
        "<bos>": "Begin of sequence", 
        "<im_patch>": "Image patch placeholder (×256)",
        "<bx_start>": "Bounding box start",
        "<bx_end>": "Bounding box end",
    }
    
    for tok, tid in special_info.items():
        purpose = purposes.get(tok, "Multimodal control")
        table_data.append([tok, str(tid), purpose])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(3):
        table[0, j].set_facecolor('#4ECDC4')
        table[0, j].set_text_props(fontweight='bold', color='white')
    
    ax.set_title("Special Tokens Overview", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "special_tokens.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_token_length_dist(tokenizer, output_dir):
    """Distribution of token lengths."""
    vocab = tokenizer.get_vocab()
    lengths = [len(tok.replace("▁", "")) for tok in vocab.keys()]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=range(0, max(lengths)+2), color='#45B7D1', edgecolor='white', alpha=0.8)
    ax.set_xlabel("Token Length (characters)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Token Length Distribution", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 25)
    
    mean_len = np.mean(lengths)
    ax.axvline(mean_len, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_length_dist.png"), dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary(tokenizer, categories, total, output_dir):
    """Generate text summary."""
    summary = []
    summary.append("=" * 60)
    summary.append("TOKENIZER & VOCABULARY OVERVIEW")
    summary.append("=" * 60)
    summary.append(f"Model: MedGemma 1.5 4B-IT + ViT3D (P2VG)")
    summary.append(f"Tokenizer Type: SentencePiece (Gemma3)")
    summary.append(f"Total Vocabulary Size: {total:,}")
    summary.append(f"Pad Token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    summary.append(f"EOS Token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    summary.append(f"BOS Token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    summary.append("")
    summary.append("Category Breakdown:")
    summary.append("-" * 40)
    for cat, tokens in categories.items():
        pct = len(tokens) / total * 100
        summary.append(f"  {cat:<25} {len(tokens):>8,} ({pct:.1f}%)")
    summary.append("")
    summary.append("Sample Medical Tokens:")
    summary.append("-" * 40)
    med_tokens = categories.get("Medical/Clinical", [])
    for t in sorted(med_tokens)[:30]:
        summary.append(f"  {t}")
    
    text = "\n".join(summary)
    print(text)
    
    with open(os.path.join(output_dir, "tokenizer_summary.txt"), "w") as f:
        f.write(text)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    print("Analyzing vocabulary...")
    categories, total = analyze_vocab(tokenizer)
    
    print("Generating visualizations...")
    plot_vocab_composition(categories, total, args.output_dir)
    plot_medical_tokens(categories, args.output_dir)
    plot_special_tokens(tokenizer, args.output_dir)
    plot_token_length_dist(tokenizer, args.output_dir)
    generate_summary(tokenizer, categories, total, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print(f"  - vocab_composition.png")
    print(f"  - medical_tokens.png")
    print(f"  - special_tokens.png")
    print(f"  - token_length_dist.png")
    print(f"  - tokenizer_summary.txt")


if __name__ == "__main__":
    main()
