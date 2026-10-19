ExecuTorch backend
==================

Torx is Mesa's ExecuTorch backend: it can make use of NPUs to accelerate ML
inference, implemented as a custom backend for the
`ExecuTorch <https://github.com/pytorch/executorch>`_ on-device AI framework.

Install ExecuTorch
------------------

Torx needs the ExecuTorch runtime library and its headers to build, and the
ExecuTorch Python package to export models. The ExecuTorch wheel provides all
of them, so there is nothing to build from source.

ExecuTorch does not pull in PyTorch, so install PyTorch too. A regular CPU-only
release is enough to export models. ExecuTorch has to be a nightly build for
now, because the nightly wheels are the ones that ship the ``executorch.pc``
pkg-config file Torx needs:

.. code-block:: console

   ~ $ pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
   ~ $ pip install --break-system-packages --pre executorch --extra-index-url https://download.pytorch.org/whl/nightly/cpu

Point pkg-config at the file the wheel ships, and check that it is found:

.. code-block:: console

   ~ $ export PKG_CONFIG_PATH=$(python3 -c 'import executorch, pathlib; print(pathlib.Path(executorch.__path__[0]) / "lib" / "pkgconfig")')
   ~ $ pkg-config --cflags --libs executorch

ExecuTorch built from source with ``-DEXECUTORCH_BUILD_SHARED=ON`` and
installed with ``cmake --install`` also provides ``executorch.pc``. Set
``PKG_CONFIG_PATH`` to its ``pkgconfig`` directory instead.

Build Mesa
----------

Build Mesa as usual, with the ``-Dtorx=true`` argument. Make sure at least one Gallium driver with NPU support is enabled.

.. code-block:: console

   ~ $ cd mesa
   mesa $ meson setup build -Dgallium-drivers=ethosu -Dvulkan-drivers= -Dtorx=true
   mesa $ meson compile -C build

Running the Test Suite
----------------------

The Torx test suite validates individual neural-network operations against CPU
references.  The models used for this testing are fetched from a separate
repository: https://gitlab.freedesktop.org/tomeu/npu-model-zoo .  At test
time, ``test_torx`` runs directly on the NPU board, executes each ``.pte`` on
the NPU, and compares the result against the CPU reference.

**Prerequisites:**

- Mesa built with ``-Dtorx=true -Dbuild-tests=true``

.. code-block:: console

   mesa $ meson configure build -Dbuild-tests=true
   mesa $ meson compile -C build

**Run:**

.. code-block:: console

   mesa $ TORX_TEST_DATA=build/src/gallium/targets/torx/tests \
          build/src/gallium/targets/torx/test_torx

To run a single test case:

.. code-block:: console

   mesa $ ... build/src/gallium/targets/torx/test_torx --gtest_filter='mobilenet_v2.042'

**Environment variables:**

``TORX_TEST_DATA``
   Directory containing pre-built test artefacts. Defaults to
   ``src/gallium/targets/torx/tests`` under the current directory.

Adding a New Model
~~~~~~~~~~~~~~~~~~

See the `NPU Model Zoo <https://gitlab.freedesktop.org/tomeu/npu-model-zoo>`_
for instructions on adding a new model to the test suite.
