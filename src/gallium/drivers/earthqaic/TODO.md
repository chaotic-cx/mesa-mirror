- Teflon: support more tensor operations and optimize existing ones (HVX).

- Teflon: subgraph parallelization, multithreaded on each NSP.

- Add a deallocator to JIT memory manager.

- Remove in-tree compiler rt builtins in favor of hexagon distro packages once available.

- Expand OpenCL/NIR support, evaluate against OpenCL CTS.

- Add mmap functionality to QShim so the driver can learn NSP virtual address mappings for device buffers.

- Add CI support.

- Improve rust: reduce unsafe, better error handling, etc.

- Add support for PIPE\_LOADER\_DEVICE\_ACCELERATOR.

- Work to stabilize and understand the qshim abstraction for Mesa usecases. Once converged on a proper understanding, migrate this functionality to the proper abstraction layers within the driver/fw stack.
