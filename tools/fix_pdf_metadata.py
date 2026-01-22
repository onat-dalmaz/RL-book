#!/usr/bin/env python3
"""
Fix PDF metadata in LaTeX file before compilation.
Adds pdftitle and pdfauthor to \hypersetup command.
"""

import re
import sys


def fix_pdf_metadata(tex_file: str) -> None:
    """Add PDF title and author metadata to LaTeX file."""
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find \hypersetup{ and add pdftitle and pdfauthor if not present
    pattern = r'(\\hypersetup\{)'
    replacement = r'\1\n      pdftitle={Assignment 1},\n      pdfauthor={Onat Dalmaz},\n'
    
    # Check if already has pdftitle
    if 'pdftitle={Assignment 1}' not in content:
        content = re.sub(pattern, replacement, content, count=1)
        
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Added PDF metadata to {tex_file}")
    else:
        print(f"PDF metadata already present in {tex_file}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.tex>")
        sys.exit(1)
    
    fix_pdf_metadata(sys.argv[1])
