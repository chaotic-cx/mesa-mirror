
# Mesa 3-D graphics library
#
# Copyright (C) 2010 LunarG Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Authors:
#    Chia-I Wu <olv@lunarg.com>

import sys
# make it possible to import glapi
import os
GLAPI = os.path.join(".", os.path.dirname(__file__), "glapi", "gen")
sys.path.insert(0, GLAPI)

from operator import attrgetter
import re
from optparse import OptionParser
import gl_XML
import glX_XML
import static_data


class ABIEntry(object):
    """Represent an ABI entry."""

    _match_c_param = re.compile(
            r'^(?P<type>[\w\s*]+?)(?P<name>\w+)(\[(?P<array>\d+)\])?$')

    def __init__(self, cols, attrs, xml_data = None):
        self._parse(cols)

        self.slot = attrs['slot']
        self.hidden = attrs['hidden']
        self.alias = attrs['alias']
        self.handcode = attrs['handcode']
        self.xml_data = xml_data

    def c_prototype(self):
        return '%s %s(%s)' % (self.c_return(), self.name, self.c_params())

    def c_return(self):
        ret = self.ret
        if not ret:
            ret = 'void'

        return ret

    def c_params(self):
        """Return the parameter list used in the entry prototype."""
        c_params = []
        for t, n, a in self.params:
            sep = '' if t.endswith('*') else ' '
            arr = '[%d]' % a if a else ''
            c_params.append(t + sep + n + arr)
        if not c_params:
            c_params.append('void')

        return ", ".join(c_params)

    def c_args(self):
        """Return the argument list used in the entry invocation."""
        c_args = []
        for t, n, a in self.params:
            c_args.append(n)

        return ", ".join(c_args)

    def _parse(self, cols):
        ret = cols.pop(0)
        if ret == 'void':
            ret = None

        name = cols.pop(0)

        params = []
        if not cols:
            raise Exception(cols)
        elif len(cols) == 1 and cols[0] == 'void':
            pass
        else:
            for val in cols:
                params.append(self._parse_param(val))

        self.ret = ret
        self.name = name
        self.params = params

    def _parse_param(self, c_param):
        m = self._match_c_param.match(c_param)
        if not m:
            raise Exception('unrecognized param ' + c_param)

        c_type = m.group('type').strip()
        c_name = m.group('name')
        c_array = m.group('array')
        c_array = int(c_array) if c_array else 0

        return (c_type, c_name, c_array)

    def __str__(self):
        return self.c_prototype()

    def __lt__(self, other):
        # compare slot, alias, and then name
        if self.slot == other.slot:
            if not self.alias:
                return True
            elif not other.alias:
                return False

            return self.name < other.name

        return self.slot < other.slot


def abi_parse_xml(xml):
    """Parse a GLAPI XML file for ABI entries."""
    api = gl_XML.parse_GL_API(xml, glX_XML.glx_item_factory())

    entry_dict = {}
    for func in api.functionIterateByOffset():
        # make sure func.name appear first
        entry_points = func.entry_points[:]
        entry_points.remove(func.name)
        entry_points.insert(0, func.name)

        for name in entry_points:
            attrs = {
                    'slot': func.offset,
                    'hidden': name not in static_data.libgl_public_functions,
                    'alias': None if name == func.name else func.name,
                    'handcode': bool(func.has_different_protocol(name)),
            }

            # post-process attrs
            if attrs['alias']:
                try:
                    alias = entry_dict[attrs['alias']]
                except KeyError:
                    raise Exception('failed to alias %s' % attrs['alias'])
                if alias.alias:
                    raise Exception('recursive alias %s' % ent.name)
                attrs['alias'] = alias
            if attrs['handcode']:
                attrs['handcode'] = func.static_glx_name(name)
            else:
                attrs['handcode'] = None

            if name in entry_dict:
                raise Exception('%s is duplicated' % (name))

            cols = []
            cols.append(func.return_type)
            cols.append(name)
            params = func.get_parameter_string(name)
            cols.extend([p.strip() for p in params.split(',')])

            ent = ABIEntry(cols, attrs, func)
            entry_dict[ent.name] = ent

    entries = sorted(entry_dict.values())

    return entries

def abi_sanity_check(entries):
    if not entries:
        return

    all_names = []
    last_slot = entries[-1].slot
    i = 0
    for slot in range(last_slot + 1):
        if entries[i].slot != slot:
            raise Exception('entries are not ordered by slots')
        if entries[i].alias:
            raise Exception('first entry of slot %d aliases %s'
                    % (slot, entries[i].alias.name))
        handcode = None
        while i < len(entries) and entries[i].slot == slot:
            ent = entries[i]
            if not handcode and ent.handcode:
                handcode = ent.handcode
            elif ent.handcode != handcode:
                raise Exception('two aliases with handcode %s != %s',
                        ent.handcode, handcode)

            if ent.name in all_names:
                raise Exception('%s is duplicated' % (ent.name))
            if ent.alias and ent.alias.name not in all_names:
                raise Exception('failed to alias %s' % (ent.alias.name))
            all_names.append(ent.name)
            i += 1
    if i < len(entries):
        raise Exception('there are %d invalid entries' % (len(entries) - 1))

class ABIPrinter(object):
    """MAPI Printer"""

    def __init__(self, entries):
        self.entries = entries

        # sort entries by their names
        self.entries_sorted_by_names = sorted(self.entries, key=attrgetter('name'))
        self.indent = ' ' * 3

    def need_entry_point(self, ent):
        """Return True if an entry point is needed for the entry."""
        # non-handcode hidden aliases may share the entry they alias
        return not (ent.hidden and ent.alias and not ent.handcode)

    def c_public_declarations(self):
        """Return the declarations of public entry points."""
        decls = []
        for ent in self.entries:
            if not self.need_entry_point(ent):
                continue
            export = 'GLAPI' if not ent.hidden else ''
            if not ent.hidden or self.is_shared:
                decls.append(self._c_decl(ent, self.prefix, True, export) + ';')

        return "\n".join(decls)

    def _c_function(self, ent, prefix, mangle=False, stringify=False):
        """Return the function name of an entry."""
        formats = { True: '"%s%s"', False: '%s%s' }
        fmt = formats[stringify]
        name = ent.name
        if mangle and ent.hidden:
            name = '_dispatch_stub_' + str(ent.name)
        return fmt % (prefix, name)

    def _c_function_call(self, ent):
        """Return the function name used for calling."""
        if ent.handcode:
            # _c_function does not handle this case
            name = '%s%s' % (self.prefix, ent.handcode)
        elif self.need_entry_point(ent):
            name = self._c_function(ent, self.prefix, True)
        else:
            name = self._c_function(ent.alias, self.prefix, True)
        return name

    def _c_decl(self, ent, prefix, mangle=False, export=''):
        """Return the C declaration for the entry."""
        decl = '%s %s %s(%s)' % (ent.c_return(), 'GLAPIENTRY',
                self._c_function(ent, prefix, mangle), ent.c_params())
        if export:
            decl = export + ' ' + decl
        return decl

    def _c_cast(self, ent):
        """Return the C cast for the entry."""
        cast = '%s (%s *)(%s)' % (
                ent.c_return(), 'GLAPIENTRY', ent.c_params())

        return cast

    def c_public_dispatches(self, no_hidden):
        """Return the public dispatch functions."""
        dispatches = []
        for ent in self.entries:
            if ent.hidden and no_hidden:
                continue

            if not self.need_entry_point(ent):
                continue

            if0 = '#if 0\n' if ent.handcode else ''
            endif = '\n#endif' if ent.handcode else ''
            decl = self._c_decl(ent, self.prefix, True, 'GLAPI' if not ent.hidden else '')
            ret = 'return ' if ent.ret else ''
            cast = self._c_cast(ent)

            dispatches.append(r"""%s%s
{
   const struct _glapi_table *_tbl = GET_DISPATCH();
   _glapi_proc _func = ((const _glapi_proc *) _tbl)[%d];
   %s((%s) _func)(%s);
}%s
""" % (if0, decl, ent.slot, ret, cast, ent.c_args(), endif))

        return '\n'.join(dispatches)

    def c_public_initializer(self):
        """Return the initializer for public dispatch functions."""
        names = []
        for ent in self.entries:
            if ent.alias:
                continue

            name = '%s(_glapi_proc) %s' % (self.indent, self._c_function_call(ent))
            names.append(name)

        return ',\n'.join(names)

    def c_stub_string_pool(self):
        """Return the string pool for use by stubs."""
        # sort entries by their names
        sorted_entries = sorted(self.entries, key=attrgetter('name'))

        pool = []
        offsets = {}
        count = 0
        for ent in sorted_entries:
            offsets[ent] = count
            pool.append('%s' % (ent.name))
            count += len(ent.name) + 1

        pool_str =  self.indent + '"' + \
                ('\\0"\n' + self.indent + '"').join(pool) + '";'
        return (pool_str, offsets)

    def c_stub_initializer(self, pool_offsets):
        """Return the initializer for struct mapi_stub array."""
        stubs = []
        for ent in self.entries_sorted_by_names:
            stubs.append('%s{ %d, %d }' % (
                self.indent, pool_offsets[ent], ent.slot))

        return ',\n'.join(stubs)

    def c_noop_functions(self):
        """Return the noop functions."""
        noops = []
        for ent in self.entries:
            if ent.alias:
                continue

            proto = self._c_decl(ent, 'noop', False, 'static')

            stmt1 = self.indent;
            space = ''
            for t, n, a in ent.params:
                stmt1 += "%s(void) %s;" % (space, n)
                space = ' '

            if ent.params:
                stmt1 += '\n';

            stmt1 += self.indent + '_mesa_noop_entrypoint(%s);' % (
                    self._c_function(ent, 'gl', False, True))

            if ent.ret:
                stmt2 = self.indent + 'return (%s) 0;' % (ent.ret)
                noop = '%s\n{\n%s\n%s\n}' % (proto, stmt1, stmt2)
            else:
                noop = '%s\n{\n%s\n}' % (proto, stmt1)

            noops.append(noop)

        return '\n\n'.join(noops)

    def c_noop_initializer(self):
        """Return an initializer for the noop dispatch table."""
        entries = [self._c_function(ent, 'noop') for ent in self.entries if not ent.alias]
        pre = self.indent + '(_glapi_proc) '
        return pre + (',\n' + pre).join(entries)

    def c_asm_gcc(self, no_hidden):
        asm = []

        for ent in self.entries:
            if ent.hidden and no_hidden:
                continue

            if not self.need_entry_point(ent):
                continue

            name = self._c_function(ent, self.prefix, True, True)

            if ent.handcode:
                asm.append('#if 0')

            if ent.hidden:
                asm.append('".hidden "%s"\\n"' % (name))

            if ent.alias and not (ent.alias.hidden and no_hidden):
                asm.append('".globl "%s"\\n"' % (name))
                asm.append('".set "%s", "%s"\\n"' % (name,
                    self._c_function(ent.alias, self.prefix, True, True)))
            else:
                asm.append('STUB_ASM_ENTRY(%s)"\\n"' % (name))
                asm.append('"\\t"STUB_ASM_CODE("%d")"\\n"' % (ent.slot))

            if ent.handcode:
                asm.append('#endif')
            asm.append('')

        return "\n".join(asm)

    def output_for_lib(self):
        print('/* This file is automatically generated by mapi_abi.py.  Do not modify. */')
        print()
        print('#include "util/glheader.h"\n')
        print()
        print('#define _gloffset_COUNT %d' % (static_data.function_count))
        print()
        print('#ifdef MAPI_TMP_DEFINES')
        print()
        print('#if defined(_WIN32) && defined(_WINDOWS_)')
        print('#error "Should not include <windows.h> here"')
        print('#endif')
        print()
        print(self.c_public_declarations())
        print('#undef MAPI_TMP_DEFINES')
        print('#endif /* MAPI_TMP_DEFINES */')

        if self.is_shared:
            print()
            print('#ifdef MAPI_TMP_NOOP_ARRAY')
            print()
            print(self.c_noop_functions())
            print()
            print('const _glapi_proc table_noop_array[] = {')
            print(self.c_noop_initializer())
            print('};')
            print()
            print('#undef MAPI_TMP_NOOP_ARRAY')
            print('#endif /* MAPI_TMP_NOOP_ARRAY */')

            pool, pool_offsets = self.c_stub_string_pool()
            print()
            print('#ifdef MAPI_TMP_PUBLIC_STUBS')
            print('static const char public_string_pool[] =')
            print(pool)
            print()
            print('static const struct mapi_stub public_stubs[] = {')
            print(self.c_stub_initializer(pool_offsets))
            print('};')
            print('#undef MAPI_TMP_PUBLIC_STUBS')
            print('#endif /* MAPI_TMP_PUBLIC_STUBS */')

            print()
            print('#ifdef MAPI_TMP_PUBLIC_ENTRIES')
            print(self.c_public_dispatches(False))
            print()
            print('static const _glapi_proc public_entries[] = {')
            print(self.c_public_initializer())
            print('};')
            print('#undef MAPI_TMP_PUBLIC_ENTRIES')
            print('#endif /* MAPI_TMP_PUBLIC_ENTRIES */')

            print()
            print('#ifdef MAPI_TMP_STUB_ASM_GCC')
            print('__asm__(')
            print(self.c_asm_gcc(False))
            print(');')
            print('#undef MAPI_TMP_STUB_ASM_GCC')
            print('#endif /* MAPI_TMP_STUB_ASM_GCC */')
        else:
            all_hidden = True
            for ent in self.entries:
                if not ent.hidden:
                    all_hidden = False
                    break
            if not all_hidden:
                print()
                print('#ifdef MAPI_TMP_PUBLIC_ENTRIES_NO_HIDDEN')
                print(self.c_public_dispatches(True))
                print()
                print('/* does not need public_entries */')
                print('#undef MAPI_TMP_PUBLIC_ENTRIES_NO_HIDDEN')
                print('#endif /* MAPI_TMP_PUBLIC_ENTRIES_NO_HIDDEN */')

                print()
                print('#ifdef MAPI_TMP_STUB_ASM_GCC_NO_HIDDEN')
                print('__asm__(')
                print(self.c_asm_gcc(True))
                print(');')
                print('#undef MAPI_TMP_STUB_ASM_GCC_NO_HIDDEN')
                print('#endif /* MAPI_TMP_STUB_ASM_GCC_NO_HIDDEN */')

class GLAPIPrinter(ABIPrinter):
    """OpenGL API Printer"""

    def __init__(self, entries):
        for ent in entries:
            self._override_for_api(ent)
        super(GLAPIPrinter, self).__init__(entries)
        self.is_shared = False
        self.prefix = 'gl'

    def _override_for_api(self, ent):
        """Override attributes of an entry if necessary for this
        printer."""
        # By default, no override is necessary.
        pass

class SharedGLAPIPrinter(GLAPIPrinter):
    """Shared GLAPI API Printer"""

    def __init__(self, entries):
        super(SharedGLAPIPrinter, self).__init__(entries)
        self.is_shared = True
        self.prefix = ''

    def _override_for_api(self, ent):
        ent.hidden = True
        ent.handcode = False

def parse_args():
    printers = ['glapi', 'shared-glapi']

    parser = OptionParser(usage='usage: %prog [options] <xml_file>')
    parser.add_option('-p', '--printer', dest='printer',
            help='printer to use: %s' % (", ".join(printers)))

    options, args = parser.parse_args()
    if not args or options.printer not in printers:
        parser.print_help()
        sys.exit(1)

    if not args[0].endswith('.xml'):
        parser.print_help()
        sys.exit(1)

    return (args[0], options)

def main():
    printers = {
        'glapi': GLAPIPrinter,
        'shared-glapi': SharedGLAPIPrinter,
    }

    filename, options = parse_args()

    entries = abi_parse_xml(filename)
    abi_sanity_check(entries)

    printer = printers[options.printer](entries)
    printer.output_for_lib()

if __name__ == '__main__':
    main()
