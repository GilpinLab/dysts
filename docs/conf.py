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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'dysts'
copyright = '2021--2025, William Gilpin'
author = 'The Gilpin Lab'

# The full version, including alpha/beta/rc tags
release = '1.0'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx.ext.napoleon'
]
# autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_index_modules = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
#html_theme = "sphinx_book_theme"

# html_theme_options = {
#     # 'github_button': True, 
# #     'github_user': 'williamgilpin',
# #     'github_repo': 'dysts',
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


#def include_only_tagged(app, what, name, obj, skip, options):
#    inclusion_tag_format = ".. only::"
#    for tag in app.tags.tags:
#        if obj.__doc__ is not None and inclusion_tag_format.format(tag) in obj.__doc__:
#            return False
#    return True
    
#def setup(app):
#    if(len(app.tags.tags)>0):
#        app.connect('autodoc-skip-member', include_only_tagged)
