"""
Sphinx configuration for Data Management GenAI package
"""

import sys
from pathlib import Path

# Add the package to the path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

# Project information
project = 'Data Management GenAI'
copyright = '2026, Yavin.O'
author = 'Yavin.O'
release = '0.1.3'
version = '0.1.3'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__'
}
autosummary_generate = True

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': False,
    'navigation_depth': 3,
    'sticky_navigation': True,
    'style_external_links': True,
    'vcs_pageview_mode': 'blob',
}

html_static_path = ['_static']
html_logo = None
html_favicon = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'snowflake': ('https://docs.snowflake.com/', None),
}

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Master document
master_doc = 'index'
