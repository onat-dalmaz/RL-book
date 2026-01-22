#!/usr/bin/env python3
"""
Sanitize Jupyter notebook for LaTeX PDF export.
Fix Unicode characters that break LaTeX and normalize math delimiters
in markdown so nbconvert keeps math in math mode.
"""

import json
import sys


def _convert_math_delimiters(text: str) -> str:
    r"""Convert \( \) and \[ \] to $ and $$ outside code blocks."""
    parts = text.split("```")
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside fenced code block; leave unchanged.
            continue
        subparts = part.split("`")
        for j, sub in enumerate(subparts):
            if j % 2 == 1:
                # Inside inline code; leave unchanged.
                continue
            sub = sub.replace(r"\[", "$$")
            sub = sub.replace(r"\]", "$$")
            sub = sub.replace(r"\(", "$")
            sub = sub.replace(r"\)", "$")
            subparts[j] = sub
        parts[i] = "`".join(subparts)
    return "```".join(parts)


def sanitize_notebook(input_path: str, output_path: str) -> None:
    """Sanitize a Jupyter notebook for LaTeX export."""
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Fix Unicode characters that actually break LaTeX compilation
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
                # Preserve the original list structure - each item is a line or part of a line
                # Join to get full text, fix Unicode, then reconstruct preserving structure
                original_text = ''.join(source)
                text = original_text
                for uchar, replacement in unicode_fixes.items():
                    text = text.replace(uchar, replacement)
                text = _convert_math_delimiters(text)
                if text != original_text:
                    # Reconstruct the list structure by splitting on newlines
                    # and preserving how the original was structured
                    lines = text.split('\n')
                    result = []
                    for i, line in enumerate(lines):
                        result.append(line)
                        # Add newline as separate item (matching Jupyter's format)
                        if i < len(lines) - 1:
                            result.append('\n')
                    # If original ended with newline, preserve that
                    if original_text.endswith('\n') and not text.endswith('\n'):
                        result.append('\n')
                    cell['source'] = result
                    changed = True
            elif isinstance(source, str):
                original = source
                for uchar, replacement in unicode_fixes.items():
                    source = source.replace(uchar, replacement)
                source = _convert_math_delimiters(source)
                if source != original:
                    cell['source'] = source
                    changed = True
    
    # Write sanitized notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    if changed:
        print(f"Sanitized notebook written to: {output_path} (Unicode + math delimiter fixes)")
    else:
        print(f"Notebook copied to: {output_path} (no changes needed)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.ipynb> <output.ipynb>")
        sys.exit(1)
    
    sanitize_notebook(sys.argv[1], sys.argv[2])
