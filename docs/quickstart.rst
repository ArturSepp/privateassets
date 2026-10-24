.. meta::
   :description: An offline privateassets quickstart: a deterministic PME calculation on
      synthetic private-asset cash flows, with no files, network access or optional extras.

.. _five-minute-quickstart:

Offline PME quickstart
======================

Author: `Artur Sepp <https://github.com/ArturSepp>`_ / First recorded:
`2026-09-08 <https://github.com/ArturSepp/privateassets/commit/e9092d6160951fab1a7ccc814d994200bcdd4e48>`_.

Implemented in `privateassets <https://github.com/ArturSepp/privateassets>`_.
Software citation: `CITATION.cff <https://github.com/ArturSepp/privateassets/blob/main/CITATION.cff>`_.

Reproducible released workflow
------------------------------

The tested support floor for this CPU workflow is **privateassets 0.7.1** on Python 3.12.
This is a verified floor, not the historical introduction version. Install core without extras;
save the existing package-owned script and run it outside synchronized folders:

.. code-block:: powershell

   python -m pip install "privateassets==0.7.1"
   python -c "from importlib.metadata import version; import privateassets as p; print(version('privateassets')); print(p.__file__)"
   Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/ArturSepp/PrivateAssets/871c7f330fbbf86d496a8f716db57b4c17cd0fa6/examples/first_success.py' -OutFile first_success.py
   python first_success.py

Installation and saving the script need internet; execution is offline. Measure installation
separately from execution and allow any first Numba compilation to finish. There is no universal
time-to-first-result promise.

The input is entirely synthetic: a contribution of 100 on 1 January 2020, a distribution of 110
on 1 January 2021, zero terminal residual NAV and a benchmark growing from 100 to 110. Cash-flow
signs are negative for contributions and positive for distributions. PME is dimensionless; this
first task does not estimate annualised alpha or factor exposures. The script checks for a
finite positive PME and prints the installed version and a success line. It writes no files,
opens no GUI and needs no vendor, client/LP records, credentials or optional extras.

The MIT core consumes qis. The optional ``factors`` extra adds FactorLasso; it is not needed
for this path. Preserve its separate dependency/licence boundary when choosing later workflows.
`Stable documentation <https://privateassets.readthedocs.io/en/stable/>`_ is the release channel;
`latest <https://privateassets.readthedocs.io/en/latest/>`_ follows main and may describe unreleased
changes. Check the build's source version rather than inferring it from a stable alias.

Run the canonical example
-------------------------

Install the core package as described in :doc:`install`. The deterministic
example below calculates PME on synthetic cash flows without files, network
access, or optional extras.

From a source checkout, run:

.. code-block:: console

   python examples/first_success.py

The script prints the installed package version followed by
``core PME calculation succeeded``. The ``examples/`` directory is not shipped
in the wheel; wheel users can copy the complete script below.

.. literalinclude:: ../examples/first_success.py
   :language: python

Read :doc:`conventions` before applying the :doc:`workflows` to empirical inputs.
