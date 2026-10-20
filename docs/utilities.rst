Development Utilities
=====================

`Mesa demos collection <https://gitlab.freedesktop.org/mesa/demos>`__
   includes several utility routines in the ``src/util/`` directory.
`Piglit <https://piglit.freedesktop.org>`__
   is an open-source test suite for OpenGL and OpenCL implementations.
`ApiTrace <https://github.com/apitrace/apitrace>`__
   is a project to trace, analyze, and debug OpenGL, Direct3D, and
   DirectDraw APIs.
Built-in apitrace tracing
   Mesa built with ``-Dapitrace=true`` builds apitrace as a subproject
   and puts its tracers in Mesa's own dispatch, so no wrapper library
   has to be loaded in front of Mesa, which makes tracing work where
   that isn't possible, like for Android apps. With
   :envvar:`MESA_APITRACE` set to 1, each process records its GL, GLES,
   EGL, and GLX calls before glthread and the driver see them, to the
   file that :envvar:`MESA_GL_TRACE_FILE` names. Calls that stock
   apitrace can't replay, like EGLImage and memory object imports, are
   recorded with equivalents that it can replay, and apps that render
   only to shared textures get presents, so the traces replay with
   ``glretrace`` and ``eglretrace``. :doc:`envvars` lists the other
   options, and :doc:`android` describes tracing Android apps.
`GFXReconstruct <https://github.com/LunarG/gfxreconstruct>`__
   is a project to trace and replay Vulkan API calls.
`Valgrind <https://valgrind.org/>`__
   is a very useful tool for tracking down memory-related problems in
   your code.
`Coverity <https://scan.coverity.com/projects/mesa>`__
   provides static code analysis of Mesa. If you create an account you
   can see the results and try to fix outstanding issues.
`gpu-ratemeter <https://gitlab.freedesktop.org/mesa/gpu-ratemeter>`__
   is a microbenchmarking tool for driver performance validation,
   performance comparisons between APIs and API codepaths, and
   characterizing the performance of undocumented GPU architectures.
   An indexable mirror is `here <https://github.com/marekolsak/gpu-ratemeter>`__.
