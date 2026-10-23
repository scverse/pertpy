#!/usr/bin/env python
# mypy: ignore-errors

import os
from datetime import datetime
from importlib.metadata import metadata

from scverse_doc.registry import intersphinx

info = metadata("pertpy")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]
release = info["Version"]
language = "en"

extensions = [
    "scverse_doc",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.load_style",
    "sphinx_remove_toctrees",
    "sphinx_tabs.tabs",
    "sphinx_issues",
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
]

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "https://pertpy.readthedocs.io/en/stable/")

ogp_site_url = html_baseurl
ogp_site_name = "pertpy"
ogp_image = "_static/pertpy_logo.png"
ogp_enable_meta_description = True
ogp_social_cards = {"enable": False}

nbsphinx_execute = "never"

bibtex_bibfiles = ["references.bib"]

suppress_warnings = ["toc.not_included"]

autosummary_imported_members = True
napoleon_google_docstring = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
myst_url_schemes = ("http", "https", "mailto")

typehints_defaults = "comma"

html_logo = "_static/pertpy_logo.svg"
html_theme_options = {"secondary_accent": True}

html_static_path = ["_static"]
html_extra_path = ["llms.txt"]
html_css_files = ["css/overwrite.css", "css/sphinx_gallery.css"]
html_show_sphinx = False

add_module_names = False
autodoc_mock_imports = ["ete4"]
intersphinx_mapping = intersphinx("annbatch") | {
    "seaborn": ("https://seaborn.pydata.org/", None),
    "pyro": ("https://docs.pyro.ai/en/stable/", None),
    "pymde": ("https://pymde.org/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "ete": ("https://etetoolkit.org/docs/latest/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}
nitpick_ignore = [
    ("py:class", "ete4.core.tree.Tree"),
    ("py:class", "ete4.treeview.TreeStyle"),
    ("py:class", "pertpy.tools._distances._distances.MeanVar"),
]

sphinx_gallery_conf = {"nested_sections=": False}
nbsphinx_thumbnails = {
    "tutorials/notebooks/guide_rna_assignment": "_static/tutorials/guide_rna_assignment.png",
    "tutorials/notebooks/perturbation_efficacy": "_static/tutorials/perturbation_efficacy.png",
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
