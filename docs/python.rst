Python reference
================

Generated from the docstrings of the installed module, which are the ones
``help(meshsearch)`` prints. Every member runs the same code the
`C++ reference <cpp.html>`__ documents, and the `guide <guide.html>`__
explains how they work together.

The signatures below show arguments as plain objects. Where a member takes a
point, ``x``, ``y`` and ``z`` are real numbers, or, for a batch, one-dimensional
array-likes of real numbers: a float64 or integer numpy array is used as it
is, anything else is converted by numpy first. Where it takes objects,
``index`` is an integer and ``indices`` a one-dimensional array-like of
integers of any dtype, never cast. An argument that has a length selects the
batched form.

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
   ``"0.3.0"``; the same as ``MESHSEARCH_VERSION`` in C++.

Errors
------

.. autoexception:: meshsearch.Error
   :show-inheritance:

.. autoexception:: meshsearch.IndexError
   :show-inheritance:
