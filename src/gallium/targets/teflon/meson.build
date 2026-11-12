libteflon = shared_library(
  'teflon',
  [ 'teflon_target.c' ],
  include_directories : [ inc_src, inc_util, inc_include, inc_gallium, inc_gallium_aux, inc_gallium_winsys, inc_gallium_drivers ],
  link_whole : [ libteflon_st ],
  link_with : [libpipe_loader_static, libws_null, libwsw, libswdri, libswkmsdri, libgallium ],
  gnu_symbol_visibility : 'hidden',
  link_args : ld_args_build_id,
  dependencies : [
    driver_etnaviv,
    driver_rocket,
    driver_ethosu,
    idep_nir,
    idep_mesautil,
  ],
  install : true,
)

if with_tests
  tensorflow_lite = shared_library(
    'tensorflow-lite-stub',
    [ 'tflite-stub.c' ],
    include_directories : [ inc_include ],
    link_args : [ '-Wl,-soname,libtensorflow-lite.so' ],
    install : false,
  )

  teflon_models_dir = meson.current_build_dir() / 'tests' / 'models'

  # This is just so modifications to the wrap file will trigger a reconfigure.
  fs.read(meson.project_source_root() / 'subprojects' / 'npu-model-zoo.wrap')

  py_flatbuffers = import('python').find_installation(
    prog_python.full_path(), modules : ['flatbuffers'], required : false)
  if py_flatbuffers.found()
    zoo = subproject('npu-model-zoo', required : get_option('wrap_mode') != 'nodownload')
    if zoo.found()
      foreach family, entry : zoo.get_variable('zoo_test_models')
        if 'tflite' not in entry['models']
          continue
        endif
        custom_target(
          'split_teflon_@0@'.format(family),
          input : entry['models']['tflite'][0],
          command : [
            prog_python, files('tests/split_tflite_tests.py'),
            'split', '@INPUT@', teflon_models_dir / family,
          ],
          output : 'split_teflon_@0@.stamp'.format(family),
          capture : true,
          build_by_default : true,
          depend_files : files('tests/split_tflite_tests.py'),
        )
      endforeach
    else
      message('teflon: the model zoo subproject is not available (wrap downloads '
              + 'disabled, or the wrap unresolved), so the model tests have no models')
    endif
  else
    message('teflon: the Python flatbuffers module is not available, so the '
            + 'model tests have no models')
  endif

  executable(
    'test_teflon',
    'test_teflon.cpp',
    'test_executor.cpp',
    'tflite-ref-ops.cc',
    dependencies : [ idep_mesautil, idep_gtest ],
    link_with : [ tensorflow_lite ],
    include_directories : [ inc_include, inc_util ],
  )
endif
