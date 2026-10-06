import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'enuma'
copyright = '2026, Yapeng Zhang'
author = 'Yapeng Zhang'
release = '0.1.0'
root_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx_automodapi.automodapi',
    # 'nbsphinx'
]

# enuma uses Google-style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

numpydoc_show_class_members = False

# :example:`fit_kelt9_timeseries.py` links a script in the repository's examples/.
extlinks = {
    "example": ("https://github.com/yapenzhang/enuma/blob/main/examples/%s", "%s"),
}

# templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# enuma's runtime dependencies are heavy/compiled (jax, numpyro, cfgrib -> eccodes).
# The default RTD build installs the package (see ``.readthedocs.yaml``), so autodoc
# imports the real modules. If that install ever becomes too heavy or fails on RTD,
# uncomment the block below and drop the ``method: pip`` install from
# ``.readthedocs.yaml`` so the docs build from source with these modules mocked out.
# autodoc_mock_imports = [
#     "jax", "jaxlib", "numpyro", "optax", "cfgrib", "xarray",
#     "s3fs", "h5netcdf", "h5py", "astropy", "scipy", "matplotlib", "tqdm",
# ]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_title = "enuma documentation"

html_theme_options = {
    "repository_url": "https://github.com/yapenzhang/enuma",
    "use_repository_button": True,
}
