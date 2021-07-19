Usage
=====

API
---

Import the pertpy API as follows:

.. code:: python

   import pertpy.api as pp

You can then access the respective modules like:

.. code:: python

   pp.pl.cool_fancy_plot()

.. contents::
    :local:
    :backlinks: none

Data
~~~~~

.. module:: pertpy.api.data
.. currentmodule:: pertpy

..
    check ehrapy to see how we actually include the modules here

Preprocessing
~~~~~~~~~~~~~

.. automodule:: pertpy.api.preprocessing
   :members:

Tools
~~~~~

.. automodule:: pertpy.api.tools
   :members:

Plotting
~~~~~~~~

.. automodule:: pertpy.api.plot
   :members:

Command-line interface
-----------------------

.. click:: pertpy.__main__:pertpy_cli
   :prog: pertpy
   :nested: full
