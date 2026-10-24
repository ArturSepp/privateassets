.. meta::
   :description: Install privateassets from PyPI, with the optional factors extra for the shrinkage
      factor-loading estimator.

Installation
============

Author: `Artur Sepp <https://github.com/ArturSepp>`_ / First recorded:
`2026-08-23 <https://github.com/ArturSepp/privateassets/commit/09b9d8b927190680ba40b6c07834ab8f714bfae4>`_.

Implemented in `privateassets <https://github.com/ArturSepp/privateassets>`_.
Software citation: `CITATION.cff <https://github.com/ArturSepp/privateassets/blob/main/CITATION.cff>`_.

Install the core package from PyPI:

.. code-block:: console

   pip install privateassets

The optional factor-loading estimator is available separately:

.. code-block:: console

   pip install "privateassets[factors]"

The current development branch requires ``factorlasso>=1.0.4.post1`` for this
extra, using the first verified MIT FactorLasso release. This requirement will
ship in the next privateassets release; older privateassets releases may permit
historical GPL FactorLasso versions. For an existing release, request the MIT
minimum explicitly:

.. code-block:: console

   pip install "privateassets[factors]" "factorlasso>=1.0.4.post1"

Core PME and deflator calculations work without FactorLasso. Package and
dependency licences and notices remain separate.

The reviewed Windows core and factors environments also include native libraries
and headers subject to the GCC Runtime Library Exception, Microsoft Visual C++
redistributable terms and NVIDIA CUDA header terms bundled with Numba. CPU
calculations do not require a CUDA backend or runtime. The packaged
``THIRD_PARTY_NOTICES.md`` retains identified OpenBLAS and LAPACK notices. This
review does not certify other versions, platforms or redistribution of an entire
environment.
