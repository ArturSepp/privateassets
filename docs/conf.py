"""Sphinx configuration for the PrivateAssets documentation."""

import sys
from pathlib import Path
import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))
PROJECT_METADATA = tomllib.loads(
    (REPOSITORY_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
)['project']

project = 'privateassets'
author = 'Artur Sepp'
copyright = '2026, Artur Sepp'
release = PROJECT_METADATA['version']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
exclude_patterns = ['_build']
autodoc_typehints = 'description'
html_theme = 'furo'
html_title = 'privateassets - multi-factor PME for private assets'
html_baseurl = 'https://privateassets.readthedocs.io/en/latest/'
html_extra_path = ['robots.txt', 'sitemap.xml']

GOOGLE_SITE_VERIFICATION = 'cddUZk3Gsd1MySw42Rwuq_rMzUDcMNkJWekObx-QS9Y'
rst_prolog = f"""
.. meta::
   :google-site-verification: {GOOGLE_SITE_VERIFICATION}
"""
