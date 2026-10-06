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


.. _data:

Data and environment
--------------------

Opacity grids
*************

*enuma* computes the transmission from precomputed cross-section grids, one HDF5
file per species. Download them from
`KEEPER <https://keeper.mpdl.mpg.de/d/923d6a75180046e68e6e/>`_ and place them in
``data/opacities/<species>.hdf5`` (e.g. ``h2o.hdf5``, ``co2.hdf5``, …).

* Available species: ``h2o``, ``co2``, ``ch4``, ``co``, ``n2o``, ``o2``, ``o3``,
  ``no``, ``no2``, ``nh3``, ``hno3``, ``ocs``, ``so2``.
* Each grid covers 300–5200 nm at a resolving power of about 10\ :sup:`6`, on a
  (pressure, temperature) grid spanning 1–10\ :sup:`5` Pa and 185–305 K.
* Every species listed  :attr:`FitConfig.species <enuma.config.FitConfig.species>`
  must have a file, so trim ``species`` to the molecules relevant to your
  wavelength range.

Wavelengths throughout *enuma* are in **nm**.

.. TODO: state whether input wavelengths must be in vacuum or air.

Data directory
**************

By default, *enuma* reads the opacity grids from ``data/opacities/`` in the
repository. The ``data/`` directory also holds the bundled MIPAS reference
atmospheres (``day.atm``, ``equ.atm``, ``ngt.atm``). To keep the data elsewhere,
either

* set the environment variable ``ENUMA_DATA_DIR`` to a directory containing
  ``opacities/`` and the ``.atm`` files, or
* point :attr:`FitConfig.opacity_dir <enuma.config.FitConfig.opacity_dir>` at the
  opacity directory for a single fit.

Downloads and cache
*******************

Some features download data on first use and cache it locally:

* the date- and site-specific GDAS atmospheric profile
  (``reference_profile_source="gdas"``), fetched from NOAA's public AWS mirror,
  which covers mid-2021 onward;
* the PHOENIX-NewEra stellar template (``stellar_enabled=True``).

Both require internet access the first time. Files are cached in
``~/.cache/enuma/`` (subdirectories ``gdas/`` and ``phoenix-newera/``), or in
``$ENUMA_CACHE`` if set, or in
:attr:`FitConfig.cache_dir <enuma.config.FitConfig.cache_dir>`. Later fits for
the same night, site or stellar parameters run offline.

Example data
************

The scripts in the ``examples/`` folder of the repository fit real spectra taken
at the VLT and Keck. The data files are not in the repository; they are in the
same `KEEPER folder <https://keeper.mpdl.mpg.de/d/923d6a75180046e68e6e/>`_ as the
opacity grids. Download them, then either put them in ``tests/`` (where
:example:`quick_start.py` looks) or edit the file path at the top of each script.
