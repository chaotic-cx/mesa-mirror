# SPDX-License-Identifier: MIT

import argparse
import os
import sys


TRACE_PREFIX = '_mesa_apitrace_'
NEXT_PREFIX = '_mesa_apitrace_next_'
ALIAS_PREFIX = '_mesa_apitrace_alias_'
REAL_PREFIX = '_mesa_real_'


parser = argparse.ArgumentParser()
parser.add_argument('apitrace')
parser.add_argument('mode', choices=['trace', 'wrap', 'rename', 'version', 'check', 'aliases'])
parser.add_argument('--egl', action='store_true')
parser.add_argument('--glx', action='store_true')
parser.add_argument('--symbols', action='append', default=[])
parser.add_argument('--mesa-xml')
parser.add_argument('--gl-aliases', action='store_true')
args = parser.parse_args()

sys.path.insert(0, os.path.join(args.apitrace, 'wrappers'))

from gltrace import GlTracer
from egltrace import EglTracer
from glxtrace import GlxTracer
from trace import Tracer
from specs.stdapi import API, Module, Void
from specs.glapi import glapi
from specs.eglapi import eglapi
from specs.glxapi import glxapi
from dispatch import function_pointer_type


ES = 'gltrace::getContext()->profile.es()'


# GLES has no DSA, so its fakes of DSA calls bind the texture for a plain call
def dsa_texture_fake(dsa, plain):
    return ('if (%s) { '
            'GLenum _target = _mesa_apitrace_texture_target(texture); '
            'GLuint _bound = _mesa_apitrace_bound_texture(_target); '
            '_fake_glBindTexture(_target, texture); '
            '_fake_%s; '
            '_fake_glBindTexture(_target, _bound); '
            '} else _fake_%s;' % (ES, plain, dsa))


EXTRA_FAKES = {
    # with the flags that Mesa gives buffers in memory objects
    'glBufferStorageMemEXT':
        'if (%s) _fake_glBufferStorageEXT(target, size, NULL, GL_DYNAMIC_STORAGE_BIT_EXT); '
        'else _fake_glBufferStorage(target, size, NULL, GL_DYNAMIC_STORAGE_BIT);' % ES,
    'glNamedBufferStorageMemEXT':
        'if (%s) { '
        'GLint _bound = 0; '
        '_glGetIntegerv(GL_COPY_WRITE_BUFFER_BINDING, &_bound); '
        '_fake_glBindBuffer(GL_COPY_WRITE_BUFFER, buffer); '
        '_fake_glBufferStorageEXT(GL_COPY_WRITE_BUFFER, size, NULL, GL_DYNAMIC_STORAGE_BIT_EXT); '
        '_fake_glBindBuffer(GL_COPY_WRITE_BUFFER, _bound); '
        '} else _fake_glNamedBufferStorage(buffer, size, NULL, GL_DYNAMIC_STORAGE_BIT);' % ES,
    'glTexStorageMem1DEXT':
        '_fake_glTexStorage1D(target, levels, internalFormat, width);',
    'glTexStorageMem2DEXT':
        '_fake_glTexStorage2D(target, levels, internalFormat, width, height);',
    'glTexStorageMem2DMultisampleEXT':
        '_fake_glTexStorage2DMultisample(target, samples, internalFormat, width, height, fixedSampleLocations);',
    'glTexStorageMem3DEXT':
        '_fake_glTexStorage3D(target, levels, internalFormat, width, height, depth);',
    'glTexStorageMem3DMultisampleEXT':
        '_fake_glTexStorage3DMultisample(target, samples, internalFormat, width, height, depth, fixedSampleLocations);',
    'glTextureStorageMem1DEXT':
        '_fake_glTextureStorage1D(texture, levels, internalFormat, width);',
    'glTextureStorageMem2DEXT': dsa_texture_fake(
        'glTextureStorage2D(texture, levels, internalFormat, width, height)',
        'glTexStorage2D(_target, levels, internalFormat, width, height)'),
    'glTextureStorageMem2DMultisampleEXT': dsa_texture_fake(
        'glTextureStorage2DMultisample(texture, samples, internalFormat, width, height, fixedSampleLocations)',
        'glTexStorage2DMultisample(_target, samples, internalFormat, width, height, fixedSampleLocations)'),
    'glTextureStorageMem3DEXT': dsa_texture_fake(
        'glTextureStorage3D(texture, levels, internalFormat, width, height, depth)',
        'glTexStorage3D(_target, levels, internalFormat, width, height, depth)'),
    'glTextureStorageMem3DMultisampleEXT': dsa_texture_fake(
        'glTextureStorage3DMultisample(texture, samples, internalFormat, width, height, depth, fixedSampleLocations)',
        'glTexStorage3DMultisample(_target, samples, internalFormat, width, height, depth, fixedSampleLocations)'),
    'glFramebufferTextureMultisampleMultiviewOVR':
        '_fake_glFramebufferTextureMultiviewOVR(target, attachment, texture, level, baseViewIndex, numViews);',
}


# hooks of the core names, which also apply to their aliases (see present_hooks)
PRESENT_HOOKS = {
    'glBindFramebuffer':
        '_mesa_apitrace_present_bind(target, framebuffer);',
    'glBindFramebufferEXT':
        '_mesa_apitrace_present_bind(target, framebuffer);',
    'glTexStorageMem2DEXT':
        '_mesa_apitrace_present_memory_objects();',
    'glTexStorageMem3DEXT':
        '_mesa_apitrace_present_memory_objects();',
    'glTextureStorageMem2DEXT':
        '_mesa_apitrace_present_memory_objects();',
    'glTextureStorageMem3DEXT':
        '_mesa_apitrace_present_memory_objects();',
    'glSignalSemaphoreEXT':
        '_mesa_apitrace_present_signal(numTextureBarriers, textures);',
}


def is_winsys(function):
    return function.name.startswith(('egl', 'glX'))


def mesa_gl_functions():
    sys.path.append(os.path.dirname(args.mesa_xml))
    import gl_XML

    return gl_XML.parse_GL_API(args.mesa_xml).functionIterateByOffset()


def gl_owners(functions):
    names = set(f.name for f in functions)
    owners = {}
    for f in mesa_gl_functions():
        known = sorted('gl' + e for e in f.entry_points if 'gl' + e in names)
        if not known:
            continue
        owner = 'gl' + f.name if 'gl' + f.name in names else known[0]
        for name in known:
            owners[name] = owner
    return owners


def present_hooks(functions):
    owners = gl_owners(functions)
    by_owner = {owners.get(name, name): hook for name, hook in PRESENT_HOOKS.items()}
    hooks = {}
    for function in functions:
        owner = owners.get(function.name)
        if owner in by_owner:
            hooks[function.name] = PRESENT_HOOKS.get(function.name, by_owner[owner])
    return hooks


def arg_names(function):
    return ', '.join(str(arg.name) for arg in function.args)


class MesaTracer(GlxTracer, EglTracer):

    getProcAddressFunctionNames = []

    fake_function_names = GlTracer.fake_function_names + [
        'glBufferStorage',
        'glBufferStorageEXT',
        'glNamedBufferStorage',
        'glBindFramebuffer',
        'glBindTexture',
        'glBlitFramebuffer',
        'glDisable',
        'glDisablei',
        'glDisableiEXT',
        'glEGLImageTargetTexture2DOES',
        'glEnable',
        'glEnablei',
        'glEnableiEXT',
        'glFramebufferTexture2D',
        'glFramebufferTextureLayer',
        'glTexStorage2D',
        'glTexSubImage2D',
        'glTextureStorage2D',
        'glTextureSubImage2D',
        'glTexStorage1D',
        'glTexStorage2DMultisample',
        'glTexStorage3D',
        'glTexStorage3DMultisample',
        'glTextureStorage1D',
        'glTextureStorage2DMultisample',
        'glTextureStorage3D',
        'glTextureStorage3DMultisample',
        'glFramebufferTextureMultiviewOVR',
    ]

    unrecorded_function_names = [
        'glEGLImageTargetTexStorageEXT',
        'glEGLImageTargetTextureStorageEXT',
    ]

    def traceFunctionDecl(self, function):
        if function.name not in self.unrecorded_function_names:
            GlTracer.traceFunctionDecl(self, function)

    swap_function_names = [
        'eglSwapBuffers',
        'eglSwapBuffersWithDamageEXT',
        'eglSwapBuffersWithDamageKHR',
        'glXSwapBuffers',
        'glXSwapBuffersMscOML',
    ]

    def isFunctionPublic(self, function):
        return False

    # skip GlTracer's _wrapProcAddress, TRACE_HEADER provides it
    def traceApi(self, api):
        self.present_hooks = present_hooks(
            [f for f in api.getAllFunctions() if not is_winsys(f)])
        Tracer.traceApi(self, api)

    def traceFunctionImplBody(self, function):
        if function.name in self.present_hooks:
            print('    ' + self.present_hooks[function.name])
        if function.name in self.swap_function_names:
            print('    _mesa_apitrace_present_real_swap();')

        if function.name == 'glEGLImageTargetTexture2DOES':
            self.emitPendingError()
            GlTracer.traceFunctionImplBody(self, function)
            self.emitImageUpload(
                'true',
                ('if (target == GL_TEXTURE_EXTERNAL_OES) {',
                 *self.externalUpload(),
                 '} else {',
                 '    _fake_glTexImage2D(target, 0, info->internalformat, info->width, info->height, 0, info->format, info->type, info->pixels);',
                 '}'),
                self.emptyUpload())
        elif function.name == 'glEGLImageTargetTexStorageEXT':
            self.emitPendingError()
            self.doInvokeFunction(function)
            # external textures as glEGLImageTargetTexture2DOES gives them
            self.emitImageUpload(
                'target == GL_TEXTURE_2D || target == GL_TEXTURE_EXTERNAL_OES',
                ('if (target == GL_TEXTURE_EXTERNAL_OES) {',
                 *self.externalUpload(bind=True),
                 '} else {',
                 '    _mesa_apitrace_fake_image_storage(target, 0, info->internalformat, info->format, info->type, info->width, info->height, info->pixels);',
                 '}'),
                ('if (target == GL_TEXTURE_EXTERNAL_OES) {',
                 '    _fake_glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, image);',
                 *self.emptyUpload(),
                 '} else {',
                 '    _mesa_apitrace_fake_image_storage(target, 0, GL_NONE, GL_NONE, GL_NONE, 0, 0, NULL);',
                 '}'))
        elif function.name == 'glEGLImageTargetTextureStorageEXT':
            self.emitPendingError()
            self.doInvokeFunction(function)
            # desktop GL only, which has no external textures
            print('    GLenum _target = _mesa_apitrace_texture_target(texture);')
            self.emitImageUpload(
                '_target == GL_TEXTURE_2D',
                ('_mesa_apitrace_fake_image_storage(_target, texture, info->internalformat, info->format, info->type, info->width, info->height, info->pixels);',),
                ('_mesa_apitrace_fake_image_storage(_target, texture, GL_NONE, GL_NONE, GL_NONE, 0, 0, NULL);',))
        elif function.name in EXTRA_FAKES:
            self.emitPendingError()
            GlTracer.traceFunctionImplBody(self, function)
            self.emitCallError()
            print('    if (_error == GL_NO_ERROR) {')
            print('        ' + EXTRA_FAKES[function.name])
            print('    }')
            self.emitSetError()
        elif function.name in ('eglDestroyImage', 'eglDestroyImageKHR'):
            print('    _mesa_apitrace_image_forget((void *)image);')
            EglTracer.traceFunctionImplBody(self, function)
        elif function.name.startswith('glX'):
            GlxTracer.traceFunctionImplBody(self, function)
        else:
            EglTracer.traceFunctionImplBody(self, function)

        if function.name in self.swap_function_names:
            print('    if (trace::localWriterFlushIntervalMs)')
            print('        trace::localWriter.sync();')

    def doInvokeFunction(self, function):
        suspend = is_winsys(function) and function.name != 'glXUseXFont'
        if suspend:
            print('    _mesa_apitrace_suspend();')
        GlTracer.doInvokeFunction(self, function)
        if suspend:
            print('    _mesa_apitrace_resume();')

    def emitPendingError(self):
        print('    GLenum _pending = _glGetError();')

    def emitCallError(self):
        print('    GLenum _error = _glGetError();')

    def emitSetError(self):
        print('    if (_pending != GL_NO_ERROR)')
        print('        _mesa_apitrace_set_error(_pending);')
        print('    else if (_error != GL_NO_ERROR)')
        print('        _mesa_apitrace_set_error(_error);')

    # the deduplicated upload to the bound external texture, which eglretrace
    # only takes after a glEGLImageTargetTexture2DOES, so fake one if needed
    def externalUpload(self, bind=False):
        return (
            *(('    _fake_glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, image);',) if bind else ()),
            '    const void *_pixels = _mesa_apitrace_image_begin((void *)image, info->width, info->height, info->pixels, info->size) ? info->pixels : NULL;',
            '    _fake_glTexImage2D(GL_TEXTURE_EXTERNAL_OES, 0, info->internalformat, info->width, info->height, 0, info->format, info->type, _pixels);',
            '    _mesa_apitrace_image_end();')

    # an upload without contents, so that a replay still has the image, of a
    # size that only images that Mesa looked up have
    def emptyUpload(self):
        return (
            '    int _width, _height;',
            '    if (_mesa_apitrace_image_size((void *)image, &_width, &_height))',
            '        _fake_glTexImage2D(target, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);')

    # the capture runs on the app's context, so set the app's errors after it;
    # missing runs when there is no capture
    def emitImageUpload(self, supported, calls, missing=()):
        self.emitCallError()
        print('    if (_error == GL_NO_ERROR) {')
        print('        gltrace::Context *_image_ctx = gltrace::getContext();')
        print('        _mesa_apitrace_quiet_debug _quiet;')
        print('        image_info *info = NULL;')
        print('        if (!(%s))' % supported)
        print(r'            os::log("apitrace: warning: %s: unsupported target\n", __FUNCTION__);')
        print('        else if (!(info = _mesa_apitrace_read_image((void *)image)))')
        print('            info = _EGLImageKHR_get_image_info(GL_TEXTURE_2D, (EGLImageKHR)image, _image_ctx->profile, _image_ctx->features);')
        # the empty upload's NULL is an offset into a bound unpack buffer too
        if missing:
            print('        gltrace::FakeUnpackScope _unpack(_image_ctx);')
        print('        if (info) {')
        if not missing:
            print('            gltrace::FakeUnpackScope _unpack(_image_ctx);')
        for call in calls:
            print('            ' + call)
        print('            _EGLImageKHR_free_image_info(info);')
        if missing:
            print('        } else {')
            for call in missing:
                print('            ' + call)
        print('        }')
        print('        (void)_glGetError();')
        print('    }')
        self.emitSetError()

    def footer(self, api):
        GlTracer.footer(self, api)

        functions = sorted(api.getAllFunctions(), key=lambda f: f.name)

        print('void')
        print('_mesa_apitrace_fake_gen_framebuffer(unsigned name)')
        print('{')
        print('    unsigned _call = trace::localWriter.beginEnter(&_glGenFramebuffers_sig, true);')
        print('    trace::localWriter.beginArg(0);')
        print('    trace::localWriter.writeSInt(1);')
        print('    trace::localWriter.endArg();')
        print('    trace::localWriter.endEnter();')
        print('    trace::localWriter.beginLeave(_call);')
        print('    trace::localWriter.beginArg(1);')
        print('    trace::localWriter.beginArray(1);')
        print('    trace::localWriter.beginElement();')
        print('    trace::localWriter.writeUInt(name);')
        print('    trace::localWriter.endElement();')
        print('    trace::localWriter.endArray();')
        print('    trace::localWriter.endArg();')
        print('    trace::localWriter.endLeave();')
        print('}')
        print()
        swap = [f for f in functions if f.name == 'eglSwapBuffers']
        print('void')
        print('_mesa_apitrace_fake_swap(void *_dpy, void *_surface)')
        print('{')
        if swap:
            print('    EGLDisplay dpy = (EGLDisplay)_dpy;')
            print('    EGLSurface surface = (EGLSurface)_surface;')
            self.fake_call(swap[0], swap[0].argNames())
        print('}')
        print()

        names = set(f.name for f in functions)

        def emit_lookups(lookups):
            for name in lookups:
                print('    if (!strcmp(name, "%s"))' % name)
                print('        return (_mesa_apitrace_proc)&%s;' % name)

        print('_mesa_apitrace_proc')
        print('_mesa_apitrace_tracer_only(const char *name, bool missing)')
        print('{')
        emit_lookups(['glNotifyMappedBufferRangeVMWX'] +
                     [name for name in self.marker_functions if name in names])
        print('    if (!missing)')
        print('        return NULL;')
        emit_lookups([name for name in self.debug_functions if name in names])
        print('    return NULL;')
        print('}')
        print()

        print('static int _mesa_apitrace_offsets[%u];' % len(functions))
        print()

        owners = gl_owners([f for f in functions if not is_winsys(f)])

        for i, function in enumerate(functions):
            if is_winsys(function):
                continue
            ret = '' if function.type is Void else 'return '
            print('static ' + function.prototype(NEXT_PREFIX + function.name))
            print('{')
            if function.name in self.debug_functions:
                # the tracer provides these when Mesa doesn't
                print('    if (_mesa_apitrace_offsets[%u] < 0)' % i)
                print('        return%s;' % ('' if function.type is Void else ' 0'))
            print('    %s((%s)_mesa_apitrace_next(_mesa_apitrace_offsets[%u]))(%s);' %
                  (ret, function_pointer_type(function), i, arg_names(function)))
            print('}')
            print()

            owner = owners.get(function.name, function.name)
            if owner == function.name:
                continue
            print('static ' + function.prototype(ALIAS_PREFIX + function.name))
            print('{')
            print('    _mesa_apitrace_proc _proc = _mesa_apitrace_current(_mesa_apitrace_offsets[%u]);' % i)
            print('    if (_proc == (_mesa_apitrace_proc)&%s%s)' % (TRACE_PREFIX, owner))
            print('        _proc = (_mesa_apitrace_proc)&%s%s;' % (TRACE_PREFIX, function.name))
            print('    %s((%s)_proc)(%s);' %
                  (ret, function_pointer_type(function), arg_names(function)))
            print('}')
            print()

        print('static const struct {')
        print('    const char *name;')
        print('    _mesa_apitrace_proc trace;')
        print('    _mesa_apitrace_proc next;')
        print('    _mesa_apitrace_proc alias;')
        print('    bool owner;')
        print('} _mesa_apitrace_funcs[%u] = {' % len(functions))
        for function in functions:
            alias = 'NULL'
            if is_winsys(function):
                next = 'NULL'
            else:
                next = '(_mesa_apitrace_proc)&%s%s' % (NEXT_PREFIX, function.name)
                if owners.get(function.name, function.name) != function.name:
                    alias = '(_mesa_apitrace_proc)&%s%s' % (ALIAS_PREFIX, function.name)
            owner = owners.get(function.name) == function.name
            print('    {"%s", (_mesa_apitrace_proc)&%s%s, %s, %s, %s},' %
                  (function.name, TRACE_PREFIX, function.name, next, alias,
                   'true' if owner else 'false'))
        print('};')
        print()
        print(TRACE_FOOTER % {'count': len(functions)})


TRACE_HEADER = r'''
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "trace_writer_local.hpp"

#include "glproc.hpp"
#include "glsize.hpp"
#include "eglsize.hpp"
#include "cxx_compat.hpp"
#include "gltrace.hpp"
#include "gltrace_arrays.hpp"

#include "main/apitrace.h"

static void *
_mesa_apitrace_wrap(const char *name, void *proc);

template <class N, class P>
static inline P
_wrapProcAddress(N procName, P procPtr)
{
    return (P)_mesa_apitrace_wrap((const char *)procName, (void *)procPtr);
}

/* keeps errors of the capture's own calls from the app's debug callback */
class _mesa_apitrace_quiet_debug
{
    bool enabled = _mesa_apitrace_debug_output_off();

public:
    ~_mesa_apitrace_quiet_debug()
    {
        if (enabled)
            _mesa_apitrace_debug_output_on();
    }
};

/* the EGLImage as Mesa reads it, in a format that keeps it exactly */
static image_info *
_mesa_apitrace_read_image(void *image)
{
    struct _mesa_apitrace_image_read read;
    if (!_mesa_apitrace_image_read(image, &read))
        return NULL;
    image_info *info = new image_info;
    info->internalformat = read.internalformat;
    info->width = read.width;
    info->height = read.height;
    info->format = read.format;
    info->type = read.type;
    info->size = read.size;
    info->pixels = read.pixels;
    return info;
}
'''


TRACE_FOOTER = r'''
static bool
_mesa_apitrace_init_offsets(void)
{
    for (unsigned i = 0; i < %(count)u; i++)
        _mesa_apitrace_offsets[i] =
            _mesa_apitrace_proc_offset(_mesa_apitrace_funcs[i].name);
    return true;
}

static void
_mesa_apitrace_ensure_offsets(void)
{
    static const bool initialized = _mesa_apitrace_init_offsets();
    (void)initialized;
}

static int
_mesa_apitrace_find(const char *name)
{
    unsigned lo = 0, hi = %(count)u;
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        int cmp = strcmp(name, _mesa_apitrace_funcs[mid].name);
        if (cmp == 0)
            return mid;
        if (cmp < 0)
            hi = mid;
        else
            lo = mid + 1;
    }
    return -1;
}

void *
_mesa_apitrace_lookup(const char *name)
{
    _mesa_apitrace_ensure_offsets();

    int i = _mesa_apitrace_find(name);
    if (i >= 0 && _mesa_apitrace_funcs[i].next) {
        if (_mesa_apitrace_offsets[i] < 0)
            return NULL;
        return (void *)_mesa_apitrace_funcs[i].next;
    }
    return (void *)_mesa_apitrace_winsys_real(name);
}

_mesa_apitrace_proc
_mesa_apitrace_tracer(const char *name)
{
    int i = _mesa_apitrace_find(name);
    if (i < 0)
        return NULL;
    if (_mesa_apitrace_funcs[i].next)
        return _mesa_apitrace_funcs[i].alias;
    return _mesa_apitrace_funcs[i].trace;
}

static void *
_mesa_apitrace_wrap(const char *name, void *proc)
{
    _mesa_apitrace_proc only = _mesa_apitrace_tracer_only(name, !proc);
    if (only)
        return (void *)only;
    if (!proc)
        return NULL;

    _mesa_apitrace_proc entry = _mesa_apitrace_tracer(name);
    return entry ? (void *)entry : proc;
}

void
_mesa_apitrace_install(_mesa_apitrace_proc *table, unsigned size)
{
    _mesa_apitrace_ensure_offsets();

    for (unsigned i = 0; i < %(count)u; i++) {
        int offset = _mesa_apitrace_offsets[i];
        if (!_mesa_apitrace_funcs[i].owner || offset < 0 ||
            (unsigned)offset >= size)
            continue;
        table[offset] = _mesa_apitrace_funcs[i].trace;
    }
}
'''


WRAP_HEADER = r'''
#include "glproc.hpp"

#include "main/apitrace.h"
#include "util/u_atomic.h"
'''


WRAP_INIT = r'''
static _mesa_apitrace_proc _mesa_apitrace_active[%(count)u];

static inline _mesa_apitrace_proc
_mesa_apitrace_get(unsigned i)
{
    _mesa_apitrace_proc proc = p_atomic_read(&_mesa_apitrace_active[i]);
    if (proc)
        return proc;
    _mesa_apitrace_winsys_init(_mesa_apitrace_names, _mesa_apitrace_real,
                               _mesa_apitrace_active, %(count)u);
    return p_atomic_read(&_mesa_apitrace_active[i]);
}
'''


def winsys_module():
    module = Module()
    if args.glx:
        module.mergeModule(glxapi)
    if args.egl:
        module.mergeModule(eglapi)
    return module


def exported_symbols():
    symbols = []
    for path in args.symbols:
        with open(path) as f:
            symbols += [line.split()[-1] for line in f if line.strip()]
    return symbols


def exported_functions():
    symbols = set(exported_symbols())
    return [f for f in winsys_module().functions if f.name in symbols]


def print_trace():
    module = winsys_module()
    module.mergeModule(glapi)
    api = API()
    api.addModule(module)

    tracer = MesaTracer()
    tracer.getProcAddressFunctionNames = []
    if args.glx:
        tracer.getProcAddressFunctionNames += GlxTracer.getProcAddressFunctionNames
    if args.egl:
        tracer.getProcAddressFunctionNames += EglTracer.getProcAddressFunctionNames

    print(TRACE_HEADER)
    for function in api.getAllFunctions():
        print('#define %s %s%s' % (function.name, TRACE_PREFIX, function.name))
    print()
    for function in api.getAllFunctions():
        print('extern "C" PRIVATE ' + function.prototype() + ';')
    print()

    tracer.traceApi(api)


def exported_gl_aliases():
    functions = dict((f.name, f) for f in glapi.functions)
    owners = gl_owners(functions.values())
    symbols = set(exported_symbols())
    aliases = []
    for name in sorted(functions):
        owner = owners.get(name, name)
        if owner != name and name in symbols:
            aliases.append((functions[name],
                            functions[owner] if owner in symbols else None))
    return aliases


def print_aliases():
    for function, owner in exported_gl_aliases():
        print(function.name)


def print_wrap():
    real = [(f, f.prototype(REAL_PREFIX + f.name), REAL_PREFIX + f.name)
            for f in exported_functions()]
    if args.gl_aliases:
        real += [(f, owner.prototype() if owner else None,
                  owner.name if owner else None)
                 for f, owner in exported_gl_aliases()]
    functions = [f for f, decl, name in real]

    print(WRAP_HEADER)
    print('extern "C" {')
    for function, decl, name in real:
        if decl:
            print(decl + ';')
    print('}')
    print()
    print('static const char *const _mesa_apitrace_names[%u] = {' % len(functions))
    for function in functions:
        print('    "%s",' % function.name)
    print('};')
    print()
    print('static const _mesa_apitrace_proc _mesa_apitrace_real[%u] = {' % len(functions))
    for function, decl, name in real:
        if name:
            print('    (_mesa_apitrace_proc)&%s,' % name)
        else:
            print('    NULL,')
    print('};')
    print(WRAP_INIT % {'count': len(functions)})

    for i, function in enumerate(functions):
        ret = '' if function.type is Void else 'return '
        print('extern "C" __attribute__((visibility("default")))')
        print(function.prototype())
        print('{')
        print('    %s((%s)_mesa_apitrace_get(%u))(%s);' %
              (ret, function_pointer_type(function), i, arg_names(function)))
        print('}')
        print()


def print_rename():
    for function in exported_functions():
        print('#define %s %s%s' % (function.name, REAL_PREFIX, function.name))


CHECK_HEADER = r'''
#include <type_traits>

#define GL_GLEXT_PROTOTYPES
#define GLX_GLXEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES

#include "util/glheader.h"
%(includes)s
#include "dispatch.h"

#undef GL_NV_coverage_sample
#include "glproc.hpp"

namespace {

template <class T>
struct abi_class {
    static constexpr int kind = std::is_floating_point<T>::value ? 1 : 2;
    static constexpr unsigned size = sizeof(T);
};

template <>
struct abi_class<void> {
    static constexpr int kind = 0;
    static constexpr unsigned size = 0;
};

template <class A, class B>
constexpr bool abi_same = abi_class<A>::kind == abi_class<B>::kind &&
                          abi_class<A>::size == abi_class<B>::size;

template <bool same_count, class F, class G>
struct sig_same_impl : std::false_type {};

template <class R1, class... A1, class R2, class... A2>
struct sig_same_impl<true, R1 (*)(A1...), R2 (*)(A2...)> :
    std::integral_constant<bool, abi_same<R1, R2> &&
                                 (abi_same<A1, A2> && ...)> {};

template <class F, class G>
struct sig_same : std::false_type {};

/* 32-bit Windows makes the calling convention part of the function type */
#define SIG_SAME(cc) \
template <class R1, class... A1, class R2, class... A2> \
struct sig_same<R1 (cc *)(A1...), R2 (cc *)(A2...)> : \
    sig_same_impl<sizeof...(A1) == sizeof...(A2), \
                  R1 (*)(A1...), R2 (*)(A2...)> {};

SIG_SAME()
#if defined(_WIN32) && !defined(_WIN64)
SIG_SAME(__stdcall)
#endif

}

#define CHECK(name, apitrace, mesa)     static_assert(sig_same<apitrace, mesa>::value, name)
'''


def print_check():
    slots = {}
    for f in mesa_gl_functions():
        for entry_point in f.entry_points:
            slots['gl' + entry_point] = f.name

    includes = []
    if args.egl:
        includes += ['#include <EGL/egl.h>', '#include <EGL/eglext.h>']
    if args.glx:
        includes += ['#include <GL/glx.h>', '#include <GL/glxext.h>']
    print(CHECK_HEADER % {'includes': '\n'.join(includes)})

    module = winsys_module()
    module.mergeModule(glapi)
    for function in module.functions:
        if is_winsys(function) or function.name not in slots:
            continue
        print('CHECK("%s", %s, _glptr_%s);' %
              (function.name, function_pointer_type(function),
               slots[function.name]))
    for function in exported_functions():
        print('CHECK("%s", %s, decltype(&%s));' %
              (function.name, function_pointer_type(function), function.name))


def print_version():
    print('{')
    print('    global:')
    for symbol in exported_symbols():
        print('        %s;' % symbol)
    print('    local:')
    print('        *;')
    print('};')


if __name__ == '__main__':
    if args.mode == 'version':
        print_version()
    elif args.mode == 'check':
        print_check()
    elif args.mode == 'aliases':
        print_aliases()
    elif args.mode == 'trace':
        print_trace()
    elif args.mode == 'wrap':
        print_wrap()
    else:
        print_rename()
