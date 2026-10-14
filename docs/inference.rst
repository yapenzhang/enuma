.. _inference:

Inference & fitting
===================

The fitting layer: the parameter registry and free/fixed partitioning, the
AutoDelta SVI runner, the user-facing fit orchestrators, and the learned stellar
template (:ref:`stellar-learned`).


Fit orchestrators
-----------------

.. automodapi:: enuma.inference.fit
   :no-inheritance-diagram:


Parameter sites
---------------

.. automodapi:: enuma.inference.sites
   :no-inheritance-diagram:


SVI optimizer
-------------

.. automodapi:: enuma.inference.optimizer
   :no-inheritance-diagram:


Instrument report
-----------------

.. automodapi:: enuma.inference.diagnostics
   :no-inheritance-diagram:


Learned stellar template
------------------------

.. automodapi:: enuma.inference.stellar_em
   :no-inheritance-diagram:

.. automodapi:: enuma.inference.stellar_gp
   :no-inheritance-diagram:


Block-tridiagonal linear algebra
--------------------------------

.. automodapi:: enuma.inference.banded
   :no-inheritance-diagram:
