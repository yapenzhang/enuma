.. _index:

Documentation for *enuma*
==========================

*enuma* is a JAX-based forward model and fitter for telluric transmission in
high-resolution spectra. The forward model computes the line-by-line transmission
through a layered atmosphere whose temperature and water-vapour profiles are
flexible and informed by meteorological data. The model is end-to-end
differentiable; fitting runs in NumPyro via SVI (Stochastic Variational Inference).


.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   method
   getstarted
   outputs

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api


Attribution
-----------

Please cite the *enuma* paper (in prep.) when the package is used in a publication.


Contact
-------

Contributions, feature requests, or bug reports are welcome through the
`Github page <https://github.com/yapenzhang/enuma>`_ or email to yazhang@mpia.de


.. * :ref:`genindex`
