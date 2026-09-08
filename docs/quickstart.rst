Five-minute quickstart
======================

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
