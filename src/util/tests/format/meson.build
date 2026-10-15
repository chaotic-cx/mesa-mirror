foreach t : ['u_format_test']
  test(t,
    executable(
      t,
      '@0@.c'.format(t),
      dependencies : idep_mesautil,
    ),
    suite : 'format',
    should_fail : meson.get_external_property('xfail', '').contains(t),
  )
endforeach
