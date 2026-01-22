#!/usr/bin/env python3
"""
Sanitize Jupyter notebook for LaTeX PDF export.
MINIMAL approach: only fix Unicode characters that break LaTeX.
Preserves ALL formatting, math delimiters, and underscores for proper Jupyter display.
"""

import json
import sys


def sanitize_notebook(input_path: str, output_path: str) -> None:
    """Sanitize a Jupyter notebook for LaTeX export - minimal changes only."""
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # ONLY fix Unicode characters that actually break LaTeX compilation
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
                # Join lines, fix Unicode, then split back into lines (preserve structure)
                text = ''.join(source)
                original = text
                for uchar, replacement in unicode_fixes.items():
                    text = text.replace(uchar, replacement)
                if text != original:
                    # Split back into lines, preserving the original line structure
                    # Each line should be a separate list item
                    lines = text.split('\n')
                    # Add back the newlines (except for the last line if it doesn't end with one)
                    result = []
                    for i, line in enumerate(lines):
                        result.append(line)
                        # Add newline after each line except the last (if original didn't end with newline)
                        if i < len(lines) - 1 or text.endswith('\n'):
                            result.append('\n')
                    cell['source'] = result
                    changed = True
            elif isinstance(source, str):
                original = source
                for uchar, replacement in unicode_fixes.items():
                    source = source.replace(uchar, replacement)
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
