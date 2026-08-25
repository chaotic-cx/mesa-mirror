# The Earthqaic Driver

Earthqaic is an experimental gallium driver for Qualcomm Cloud AI accelerators. It currently implements a limited backend for Teflon and OpenCL.

## Building

Earthqaic is primarily written in rust, and therefore makes use of various submodules. Meson must be configured to use the submodules, even if the libraries already exist on the system, by forcing fallback.

The driver also relies on rusticl and LLVM.

To build the driver, configure your build directory as follows (along with any other desired meson options):

```
meson setup build -Dgallium-drivers=earthqaic_experimental -Dgallium-rusticl=true
```
## Running the driver

Build objects needed at runtime, such as the QShim, are stored in the configured build directory (e.g., ```build/src/gallium/drivers/earthqaic```), and are read by the driver on the fly. If that build directory no longer exists (e.g. only an installed copy remains), the driver falls back to the installed location under `datadir` (e.g. ```<prefix>/share/earthqaic```).

To point the driver at a different location entirely (e.g. these artifacts were copied to a different path or a different machine), set:

```export EARTHQAIC_PATH=<path_to_artifacts>```

This overrides both the build-dir and installed-dir lookups above.

The driver can be run with debug output by setting the EARTHQAIC_DEBUG environment variable: ```EARTHQAIC_DEBUG=true```. Unset the variable to run without debug info.

