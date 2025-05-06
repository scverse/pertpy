#!/usr/bin/env python
# mypy: ignore-errors

import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / "extensions")]

needs_sphinx = "4.3"

info = metadata("pertpy")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]
release = info["Version"]
github_repo = "pertpy"
master_doc = "index"
language = "en"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "sphinx_remove_toctrees",
    "sphinx_design",
    "sphinx_issues",
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
]

ogp_site_url = "https://pertpy.readthedocs.io/en/latest/"
ogp_image = "https://pertpy.readthedocs.io/en/latest/_static/pertpy_logo.png"

# nbsphinx specific settings
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_*/**.ipynb",
    "auto_*/**.md5",
    "auto_*/**.py",
    "**.ipynb_checkpoints",
]
nbsphinx_execute = "never"
pygments_style = "sphinx"

templates_path = ["_templates"]
bibtex_bibfiles = ["references.bib"]
nitpicky = True  # Warn about broken links
# source_suffix = ".md"

suppress_warnings = ["toc.not_included"]

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
numpydoc_show_class_members = False
annotate_defaults = True
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
nb_execution_mode = "off"
warn_as_error = True

typehints_defaults = "comma"

# html_show_sourcelink = True
html_theme = "scanpydoc"
html_title = "pertpy"
html_logo = "_static/pertpy_logo.svg"

html_static_path = ["_static"]
html_css_files = ["css/override.css", "css/sphinx_gallery.css"]
html_show_sphinx = False

add_module_names = False
autodoc_mock_imports = ["ete4"]
intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "mudata": ("https://mudata.readthedocs.io/en/stable/", None),
    "scvi-tools": ("https://docs.scvi-tools.org/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/main", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "pyro": ("https://docs.pyro.ai/en/stable/", None),
    "pymde": ("https://pymde.org/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "ete": ("https://etetoolkit.org/docs/latest/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
}
nitpick_ignore = [
    ("py:class", "ete4.core.tree.Tree"),
    ("py:class", "ete4.treeview.TreeStyle"),
    ("py:class", "pertpy.tools._distances._distances.MeanVar"),
    ("py:class", "The requested data as a NumPy array. [ref.class]"),
    ("py:class", "The full registry saved with the model [ref.class]"),
    ("py:class", "Model with loaded state dictionaries. [ref.class]"),
]

sphinx_gallery_conf = {"nested_sections=": False}
nbsphinx_thumbnails = {
    "tutorials/notebooks/guide_rna_assignment": "_static/tutorials/guide_rna_assignment.png",
    "tutorials/notebooks/mixscape": "_static/tutorials/mixscape.png",
    "tutorials/notebooks/augur": "_static/tutorials/augur.png",
    "tutorials/notebooks/sccoda": "_static/tutorials/sccoda.png",
    "tutorials/notebooks/sccoda_extended": "_static/tutorials/sccoda_extended.png",
    "tutorials/notebooks/tasccoda": "_static/tutorials/tasccoda.png",
    "tutorials/notebooks/milo": "_static/tutorials/milo.png",
    "tutorials/notebooks/dialogue": "_static/tutorials/dialogue.png",
    "tutorials/notebooks/enrichment": "_static/tutorials/enrichment.png",
    "tutorials/notebooks/distances": "_static/tutorials/distances.png",
    "tutorials/notebooks/distance_tests": "_static/tutorials/distances_tests.png",
    "tutorials/notebooks/cinemaot": "_static/tutorials/cinemaot.png",
    "tutorials/notebooks/scgen_perturbation_prediction": "_static/tutorials/scgen_perturbation_prediction.png",
    "tutorials/notebooks/perturbation_space": "_static/tutorials/perturbation_space.png",
    "tutorials/notebooks/differential_gene_expression": "_static/tutorials/dge.png",
    "tutorials/notebooks/metadata_annotation": "_static/tutorials/metadata.png",
    "tutorials/notebooks/ontology_mapping": "_static/tutorials/ontology.png",
    "tutorials/notebooks/norman_use_case": "_static/tutorials/norman.png",
    "tutorials/notebooks/mcfarland_use_case": "_static/tutorials/mcfarland.png",
    "tutorials/notebooks/zhang_use_case": "_static/tutorials/zhang.png",
}
