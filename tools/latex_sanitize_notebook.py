#!/usr/bin/env python3
"""
Sanitize Jupyter notebook for LaTeX PDF export.
Fixes Unicode characters and math delimiters that break LaTeX compilation.
"""

import json
import re
import sys
from pathlib import Path


def sanitize_markdown(text: str) -> str:
    """Sanitize markdown text for LaTeX compatibility."""
    if not isinstance(text, str):
        return text
    
    # Step 1A: Unicode sanitization
    replacements = {
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
    
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Step 1B: Skip math delimiter conversion - nbconvert handles \( \) and \[ \] correctly
    # We don't need to convert them to $ $ or $$ $$ as this makes the notebook look weird in Jupyter
    # and nbconvert can handle the original delimiters fine
    
    # Step 1C: Escape underscores in plain text (outside code and math)
    # Split by code blocks and math regions
    result = []
    # Pattern to match code blocks, inline math \(...\) or $...$, and display math \[...\] or $$...$$
    split_pattern = re.compile(r'```[\s\S]*?```|`[^`\n]+`|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$|\\\([^\\\)]*?\\\)|\$[^$\n]+\$')
    last_pos = 0
    
    for match in split_pattern.finditer(text):
        # Add text before match with escaped underscores
        if match.start() > last_pos:
            text_segment = text[last_pos:match.start()]
            text_segment = text_segment.replace('_', r'\_')
            result.append(text_segment)
        # Add match as-is (code or math)
        result.append(match.group())
        last_pos = match.end()
    
    # Add remaining text with escaped underscores
    if last_pos < len(text):
        text_segment = text[last_pos:]
        text_segment = text_segment.replace('_', r'\_')
        result.append(text_segment)
    
    return ''.join(result)


def sanitize_notebook(input_path: str, output_path: str) -> None:
    """Sanitize a Jupyter notebook for LaTeX export."""
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Process all markdown cells
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if isinstance(source, list):
                # Join list of strings, sanitize, then split back
                text = ''.join(source)
                sanitized = sanitize_markdown(text)
                cell['source'] = list(sanitized)
            elif isinstance(source, str):
                cell['source'] = sanitize_markdown(source)
    
    # Write sanitized notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"Sanitized notebook written to: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.ipynb> <output.ipynb>")
        sys.exit(1)
    
    sanitize_notebook(sys.argv[1], sys.argv[2])
