.. _installation:

Installation
============


Installation from source
------------------------

*enuma* requires Python 3.12+. Clone the repository:

.. code-block:: console

    $ git clone https://github.com/yapenzhang/enuma.git

Then install the package by running ``pip`` in the local repository folder:

.. code-block:: console

    $ cd enuma
    $ pip install -e .


GPU support
-----------

To run the model on a GPU, manually install the CUDA-enabled build of JAX for
your system's CUDA version (e.g. CUDA 13):

.. code-block:: console

    $ pip install -U "jax[cuda13]"


Opacity data
------------

The opacity files can be downloaded from `KEEPER <https://keeper.mpdl.mpg.de/d/923d6a75180046e68e6e/>`_ 
and placed in ``data/opacities/<species>.hdf5`` (e.g. ``h2o.hdf5``, ``co2.hdf5``, …). 
