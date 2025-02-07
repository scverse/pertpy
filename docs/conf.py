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


extensions = [
    "myst_parser",
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
]

# remove_from_toctrees = ["tutorials/notebooks/*", "api/reference/*"]

# for sharing urls with nice info
ogp_site_url = "https://pertpy.readthedocs.io/en/latest/"
ogp_image = "https://pertpy.readthedocs.io/en/latest//_static/logo.png"

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

templates_path = ["_templates"]
bibtex_bibfiles = ["references.bib"]
nitpicky = True  # Warn about broken links
# source_suffix = ".md"

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True  # for pytorch lightning
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
numpydoc_show_class_members = False
annotate_defaults = True  # scanpydoc option, look into why we need this
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

# The master toctree document.
master_doc = "index"

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable/", None),
    "pyro": ("http://docs.pyro.ai/en/stable/", None),
    "pymde": ("https://pymde.org/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

language = "en"

typehints_defaults = "comma"

pygments_style = "default"
pygments_dark_style = "native"


# -- Options for HTML output -------------------------------------------

# html_show_sourcelink = True
html_theme = "furo"

# Set link name generated in the top bar.
html_title = "pertpy"
html_logo = "_static/pertpy_logos/pertpy_pure.png"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262per",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/override.css", "css/sphinx_gallery.css"]
html_show_sphinx = False

sphinx_gallery_conf = {"nested_sections=": False}


nbsphinx_prolog = r"""
.. raw:: html

{{% set docname = env.doc2path(env.docname, base=None).split("/")[-1] %}}

.. raw:: html

    <style>
        p {{
            margin-bottom: 0.5rem;
        }}
        /* Main index page overview cards */
        /* https://github.com/spatialaudio/nbsphinx/pull/635/files */
        .jp-RenderedHTMLCommon table,
        div.rendered_html table {{
        border: none;
        border-collapse: collapse;
        border-spacing: 0;
        font-size: 12px;
        table-layout: fixed;
        color: inherit;
        }}

        body:not([data-theme=light]) .jp-RenderedHTMLCommon tbody tr:nth-child(odd),
        body:not([data-theme=light]) div.rendered_html tbody tr:nth-child(odd) {{
        background: rgba(255, 255, 255, .1);
        }}
    </style>

.. raw:: html

    <div class="admonition note">
        <p class="admonition-title">Note</p>
        <p>
        This page was generated from
        <a class="reference external" href="https://github.com/scverse/pertpy/tree/{version}/">{docname}</a>.
        Some tutorial content may look better in light mode.
        </p>
    </div>
""".format(version=version, docname="{{ docname|e }}")
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
