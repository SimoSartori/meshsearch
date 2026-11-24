Python reference
================

Generated from the docstrings of the installed module, which are the ones
``help(meshsearch)`` prints. Every entry here is the same code the C++
reference documents, reached through nanobind.

.. module:: meshsearch

The grid
--------

.. autoclass:: meshsearch.MeshGrid
   :members:
   :special-members: __init__, __len__, __repr__, __copy__, __deepcopy__

The result of a batched ``close_objects``
-----------------------------------------

.. autoclass:: meshsearch.Batch
   :members: indices, offsets

The version
-----------

.. autodata:: meshsearch.__version__
   :no-value:

   The version of the library the module was built from, a string such as
   ``"1.0.0"``; the same as ``MESHSEARCH_VERSION`` in C++.

Errors
------

.. autoexception:: meshsearch.Error
   :show-inheritance:

.. autoexception:: meshsearch.IndexError
   :show-inheritance:
