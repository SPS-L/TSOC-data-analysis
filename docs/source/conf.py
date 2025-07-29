# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'TSOC Data Analysis'
copyright = '2025, Sustainable Power Systems Lab (SPSL)'
author = 'Sustainable Power Systems Lab (SPSL)'
contact = 'info@sps-lab.org'

# The full version, including alpha/beta/rc tags
# Automatically read version from __init__.py
import re


# Import the package to get version
import tsoc_data_analysis

# Extract version from __init__.py
release = tsoc_data_analysis.__version__
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'TSOCDataAnalysisdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '11pt',

    # Additional stuff for the LaTeX preamble.
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{graphicx}
        \usepackage{hyperref}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{array}
        \usepackage{multirow}
        \usepackage{wrapfig}
        \usepackage{float}
        \usepackage{colortbl}
        \usepackage{pdflscape}
        \usepackage{tabu}
        \usepackage{threeparttable}
        \usepackage{threeparttablex}
        \usepackage{makecell}
        \usepackage{xcolor}
        \usepackage{newunicodechar}
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}
        \newunicodechar{≤}{\ensuremath{\leq}}
        \newunicodechar{≥}{\ensuremath{\geq}}
        \newunicodechar{±}{\ensuremath{\pm}}
        \newunicodechar{°}{\ensuremath{^\circ}}
        \newunicodechar{μ}{\ensuremath{\mu}}
        \newunicodechar{α}{\ensuremath{\alpha}}
        \newunicodechar{β}{\ensuremath{\beta}}
        \newunicodechar{γ}{\ensuremath{\gamma}}
        \newunicodechar{δ}{\ensuremath{\delta}}
        \newunicodechar{ε}{\ensuremath{\varepsilon}}
        \newunicodechar{θ}{\ensuremath{\theta}}
        \newunicodechar{λ}{\ensuremath{\lambda}}
        \newunicodechar{π}{\ensuremath{\pi}}
        \newunicodechar{σ}{\ensuremath{\sigma}}
        \newunicodechar{φ}{\ensuremath{\phi}}
        \newunicodechar{ω}{\ensuremath{\omega}}
    ''',

    # Latex figure (float) alignment
    'figure_align': 'htbp',
    
    # Additional options for better PDF output
    'extraclassoptions': 'openany,oneside',
    'babel': '\\usepackage[english]{babel}',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'TSOCDataAnalysis.tex', 'TSOC Data Analysis Documentation',
     'Sustainable Power Systems Lab (SPSL)', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'tsocdataanalysis', 'TSOC Data Analysis Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'TSOCDataAnalysis', 'TSOC Data Analysis Documentation',
     author, 'TSOCDataAnalysis', 'A comprehensive Python tool for analyzing TSOC power system operational data from Excel files.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Autodoc configuration --------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = 'description'

# Don't show class signature with the class' name.
autodoc_class_signature = 'separated'

# Default options for autodoc directives. They are applied to all
# autodoc directives automatically.
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
} 