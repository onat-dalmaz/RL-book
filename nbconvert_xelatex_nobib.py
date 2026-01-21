"""
nbconvert configuration to use XeLaTeX and skip BibTeX.
"""

from traitlets.config import Config

c = Config()

# Force XeLaTeX
c.PDFExporter.latex_command = ['xelatex', '{filename}', '-interaction=nonstopmode', '-quiet']

# Disable BibTeX (use no-op command)
c.PDFExporter.bib_command = ['true']

# Use classic template for reliability
c.PDFExporter.template_name = 'classic'

# Additional settings for robustness
c.PDFExporter.verbose = False
