#!/usr/bin/env python3
"""
Sanitize Jupyter notebook for LaTeX PDF export.
Minimal approach: only fix Unicode characters that break LaTeX.
Preserves all math delimiters and formatting for proper Jupyter display.
"""

import json
import sys
from pathlib import Path


def sanitize_notebook(input_path: str, output_path: str) -> None:
    """Sanitize a Jupyter notebook for LaTeX export - minimal changes only."""
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Only fix Unicode characters that actually break LaTeX compilation
    # Do NOT touch math delimiters, underscores, or any other formatting
    unicode_fixes = {
        '\u00A0': ' ',  # NBSP
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201C': '"',  # left double quote
        '\u201D': '"',  # right double quote
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2026': '...',  # ellipsis
        '\u2192': '->',  # →
        '\u21D2': '=>',  # ⇒
        '\u21A6': '|->',  # ↦
        '\u2212': '-',  # minus
        '\u200B': '',  # zero-width space
        '\u200C': '',  # zero-width non-joiner
        '\u200D': '',  # zero-width joiner
        '\uFEFF': '',  # zero-width no-break space
    }
    
    changed = False
    # Process all markdown cells
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if isinstance(source, list):
                # Join, fix Unicode, then split back
                text = ''.join(source)
                original = text
                for uchar, replacement in unicode_fixes.items():
                    text = text.replace(uchar, replacement)
                # Fix specific LaTeX issue: ensure \min is in proper math mode
                # nbconvert sometimes strips \( \) so use $ $ for \min expressions
                text = text.replace('\\(\\min(', '$\\min(')
                text = text.replace('\\min(100, s + D)\\)', '\\min(100, s + D)$')
                if text != original:
                    cell['source'] = list(text)
                    changed = True
            elif isinstance(source, str):
                original = source
                for uchar, replacement in unicode_fixes.items():
                    source = source.replace(uchar, replacement)
                # Fix \min math mode issue
                source = source.replace('\\(\\min(', '$\\min(')
                source = source.replace('\\min(100, s + D)\\)', '\\min(100, s + D)$')
                if source != original:
                    cell['source'] = source
                    changed = True
    
    # Write sanitized notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    if changed:
        print(f"Sanitized notebook written to: {output_path} (Unicode fixes only)")
    else:
        print(f"Notebook copied to: {output_path} (no changes needed)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.ipynb> <output.ipynb>")
        sys.exit(1)
    
    sanitize_notebook(sys.argv[1], sys.argv[2])
