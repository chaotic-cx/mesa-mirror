#
# Copyright (C) 2018 Valve Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import unittest
import functools
import itertools

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from nir_algebraic import SearchAndReplace, AlgebraicPass

# These tests check that the bitsize validator correctly rejects various
# different kinds of malformed expressions, and documents what the error
# message looks like.

a = 'a'
b = 'b'
c = 'c'

class ValidatorTests(unittest.TestCase):
    pattern = ()
    message = ''

    algebraic_pass = AlgebraicPass("test", [])

    def common(self, pattern, message):
        with self.assertRaises(AssertionError) as context:
            SearchAndReplace(pattern, self.algebraic_pass)

        self.assertEqual(message, str(context.exception))

    def test_wrong_src_count(self):
        self.common((('iadd', a), ('fadd', a, a)),
            "Expression ('iadd', 'a') has 1 sources, expected 2")

    def test_var_bitsize(self):
        self.common((('iadd', 'a@32', 'a@64'), ('fadd', a, a)),
            "Variable a has conflicting bit size requirements: " \
            "it must have bit size 32 and 64")

    def test_var_bitsize_2(self):
        self.common((('iadd', a, 'a@32'), ('fadd', 'a@64', a)),
            "Variable a has conflicting bit size requirements: " \
            "it must have bit size 32 and 64")

    def test_search_src_bitsize(self):
        self.common((('iadd', 'a@32', 'b@64'), ('fadd', a, b)),
            "Source a@32 of ('iadd', 'a@32', 'b@64') must have bit size 32, " \
            "while source b@64 must have incompatible bit size 64")

    def test_replace_src_bitsize(self):
        self.common((('iadd', a, ('b2i', b)), ('iadd', a, b)),
            "Sources a (bit size of a) and b (bit size of b) " \
            "of ('iadd', 'a', 'b') may not have the same bit size " \
            "when building the replacement expression.")

    def test_search_src_bitsize_fixed(self):
        self.common((('ishl', a, 'b@64'), ('ishl', a, b)),
            "b@64 must have 64 bits, but as a source of nir_op_ishl " \
            "it must have 32 bits")

    def test_replace_src_bitsize_fixed(self):
        self.common((('iadd', a, b), ('ishl', a, b)),
            "b has the bit size of b, but as a source of nir_op_ishl " \
            "it must have 32 bits, which may not be the same")

    def test_search_dst_bitsize(self):
        self.common((('iadd@32', 'a@64', b), ('iadd', a, b)),
            "('iadd@32', 'a@64', 'b') must have the bit size of 32, " \
            "while its source a@64 must have incompatible bit size 64")

    def test_replace_dst_bitsize(self):
        self.common((('iadd', a, b), ('iadd@32', a, b)),
            "('iadd@32', 'a', 'b') must have 32 bits, but its source a " \
            "(bit size of b) may not have that bit size when building " \
            "the replacement.")

    def test_search_dst_bitsize_fixed(self):
        self.common((('ufind_msb@64', a), ('ineg', a)),
            "('ufind_msb@64', 'a') must have 64 bits, "\
            "but as a destination of nir_op_ufind_msb it must have 32 bits")

    def test_replace_dst_bitsize_fixed(self):
        self.common((('ineg', 'a@64'), ('ufind_msb@64', a)),
            "('ufind_msb@64', 'a') must have 64 bits, " \
            "but as a destination of nir_op_ufind_msb it must have 32 bits")

    def test_ambiguous_bitsize(self):
        self.common((('ineg', 'a@32'), ('i2b', ('b2i', a))),
            "Ambiguous bit size for replacement value ('b2i', 'a'): it "\
            "cannot be deduced from a variable, a fixed bit size somewhere, "
            "or the search expression.")

    def test_search_replace_mismatch(self):
        self.common((('b2i', ('i2b', a)), a),
            "The search expression bit size ('b2i', ('i2b', 'a')) and " \
            "replace expression bit size a may not be the same")

class OperandFilterTests(unittest.TestCase):
    iadd_ineg = (('iadd', ('ineg', a), 0), a)
    ffma_fneg = (('ffma', ('fneg', a), b, ('fabs', c)), a)
    i2i32_ineg = (('i2i32', ('ineg', a)), 'a@32')

    patterns = [
        (('bcsel', ('ilt', a, 0), ('ineg', b), c), b),
        (('bcsel', a, ('iadd', b, 0), ('imul', b, c)), b),
        iadd_ineg,
        iadd_ineg,  # Candidate order, including duplicates.
        ffma_fneg,
        i2i32_ineg,
        (('i2i', ('iadd', a, 0)), ('i2i', a)),
        (('iadd', ('imul', a, b), '#c'), a),
        (('bitfield_insert', ('ineg', a), b, 0, '#d'), a),
    ]

    def setUp(self):
        self.algebraic_pass = AlgebraicPass('test', self.patterns)
        self.automaton = self.algebraic_pass.automaton

    def pattern_indices(self, pattern):
        """Return the numbers the pass gave each copy of pattern."""
        return [i for i, p in enumerate(self.patterns) if p is pattern]

    @staticmethod
    def state_pattern_indices(automaton, state):
        return [p for item in automaton.states[state] for p in item.patterns]

    @staticmethod
    def transition(automaton, op, states):
        return automaton.table[op][tuple(automaton.filter[op, i][s]
                                         for i, s in enumerate(states))]

    @staticmethod
    def emitted_transition(src_filters, flat, states):
        # Mirrors the mixed-radix index computed by nir_algebraic_automaton().
        index = 0
        for (filt, dimension), state in zip(src_filters, states):
            index = index * dimension + (filt[state] if filt is not None else 0)
        return flat[index]

    def check_emitted_layout(self, op, src_filters, flat):
        automaton = self.automaton
        # The flat table must be exactly the product of the per-source
        # dimensions, otherwise the mixed-radix index would run off the end
        # or leave unreachable entries.
        self.assertEqual(len(flat), functools.reduce(
            lambda x, y: x * y, (n for _, n in src_filters)))
        for states in itertools.product(range(len(automaton.states)),
                                        repeat=len(src_filters)):
            # Looking up unfiltered states through the emitted filters and
            # flat table must give the same next state as the automaton's
            # own per-source table.
            self.assertEqual(self.emitted_transition(src_filters, flat, states),
                             self.transition(automaton, op, states))

    def test_rectangular_domains_and_indexing(self):
        automaton = self.automaton
        dimensions = [len(automaton.rep['bcsel', i]) for i in range(3)]
        # The test patterns put different things in each of bcsel's three
        # inputs, so each input gets a lookup of a different size. If they
        # were all the same size, the table would be a cube and this test
        # wouldn't check the case where the sizes differ.
        self.assertGreater(len(set(dimensions)), 1)
        for op in automaton.opcodes:
            self.check_emitted_layout(op, automaton.src_filters[op],
                                      automaton.op_table[op])
        rendered = self.algebraic_pass.render()
        # The C template must emit the per-source filter array form that
        # nir_algebraic_automaton() expects.
        self.assertIn('.srcs = (const struct per_src_filter []) {', rendered)
        # Each distinct nonzero filter is emitted once, including shared
        # commutative filters. Zero-only filters use NULL instead.
        filters = {tuple(filt) for op in automaton.opcodes
                   for filt, _ in automaton.src_filters[op] if filt is not None}
        self.assertEqual(rendered.count('static const uint8_t test_filter_'), len(filters))

    def test_commutative_aliases_and_conversion(self):
        automaton = self.automaton
        iadd_ineg = self.pattern_indices(self.iadd_ineg)
        self.assertEqual(len(iadd_ineg), 2)
        neg = self.transition(automaton, 'ineg', [0])
        for sources in [(neg, 1), (1, neg)]:
            state = self.transition(automaton, 'iadd', sources)
            candidates = list(itertools.takewhile(lambda p: p is not None,
                automaton.state_patterns[automaton.state_pattern_offsets[state]:]))
            # iadd(-a, 0) and iadd(0, -a) are the same addition written in
            # either order, so both should be offered the iadd(-a, 0) pattern
            # as a possible match. The pattern is listed twice, and both
            # copies should still be offered, in order.
            self.assertEqual([p for p in candidates if p in iadd_ineg], iadd_ineg)
        fneg = self.transition(automaton, 'fneg', [0])
        fabs = self.transition(automaton, 'fabs', [0])
        for sources in [(fneg, 0, fabs), (0, fneg, fabs)]:
            state = self.transition(automaton, 'ffma', sources)
            # ffma(x, y, z) computes x * y + z, and the multiply gives the same
            # result whichever way round x and y are. So the pattern with the
            # fneg in the first slot should still match when the fneg turns up
            # in the second slot instead.
            self.assertIn(self.pattern_indices(self.ffma_fneg)[0],
                          self.state_pattern_indices(automaton, state))
        # Because the first two ffma inputs can be swapped, anything that can
        # match in one of them can also match in the other, so the generator
        # gives both the same lookup table (filter). The third input can't be
        # swapped, so it gets its own.
        self.assertEqual(automaton.filter['ffma', 0], automaton.filter['ffma', 1])
        self.assertNotEqual(automaton.filter['ffma', 0], automaton.filter['ffma', 2])
        # Sized conversions are folded into their unsized opcode, which has a
        # single source.
        self.assertNotIn('i2i32', automaton.opcodes)
        state = self.transition(automaton, 'i2i', [neg])
        self.assertIn(self.pattern_indices(self.i2i32_ineg)[0],
                      self.state_pattern_indices(automaton, state))


unittest.main()
