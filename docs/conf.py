# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("drvi-py")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "theislab",
    "github_repo": "drvi",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    # General
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    # Numerical
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    # Deep learning
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    # Special
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "mudata": ("https://mudata.readthedocs.io/en/stable/", None),
    "scvi-tools": ("https://docs.scvi-tools.org/en/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project_name

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
    "use_source_button": True,
}

pygments_style = "default"

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
    # PyTorch references that are not properly resolved
    ("py:class", "Module"),
    ("py:class", "torch.nn.modules.Module"),
    ("py:class", "torch.nn.Parameter"),
    ("py:class", "Tensor"),
    ("py:class", "optional"),
    ("py:class", "torch.utils.hooks.RemovableHandle"),
    ("py:class", "Dropout"),
    ("py:class", "BatchNorm"),
    ("py:class", "Parameter"),
    ("py:attr", "state_dict"),
    ("py:attr", "strict"),
    ("py:attr", "assign"),
    ("py:attr", "persistent"),
    ("py:attr", "grad_input"),
    ("py:attr", "grad_output"),
    ("py:attr", "requires_grad"),
    ("py:attr", "dst_type"),
    ("py:attr", "dtype"),
    ("py:attr", "device"),
    ("py:attr", "non_blocking"),
    ("py:func", "add_module"),
    ("py:func", "register_module_forward_hook"),
    ("py:func", "register_module_forward_pre_hook"),
    ("py:func", "register_module_full_backward_hook"),
    ("py:func", "register_module_full_backward_pre_hook"),
    # DRVI internal references
    ("py:class", "drvi.scvi_tools_based.merlin_data._data.MerlinData"),
    ("py:class", "drvi.nn_modules.layer.factory.LayerFactory"),
]
