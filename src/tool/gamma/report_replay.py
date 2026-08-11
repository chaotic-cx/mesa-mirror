#!/usr/bin/env python3

# Copyright © 2026 Valve Corporation
# SPDX-License-Identifier: MIT

import csv
import sys
import math
import os

COLOR_RED = '\033[31m'
COLOR_GREEN = '\033[32m'
COLOR_RESET = '\033[0m'

def uncertain_number_to_str(x, u):
   if u == 0:
      return str(x)

   digits = max(1 - math.floor(math.log10(u)), 0)
   uncertainty_digits = math.floor(u * 10**digits)
   return '{:.{digits}f}({:.0f})'.format(x, uncertainty_digits, digits=digits)

before_times = dict()
before_sizes = dict()
after_times = dict()
after_sizes = dict()

with open(sys.argv[1]) as before_file:
    reader = csv.DictReader(before_file)
    for row in reader:
        selector = (os.path.basename(row['Filename']), int(row['ASIndex']))
        if selector not in before_times:
            before_times[selector] = []
        if selector not in before_sizes:
            before_sizes[selector] = []
        before_times[selector].append(float(row['BuildTime']))
        before_sizes[selector].append(float(row['CompactedSize']))

with open(sys.argv[2]) as after_file:
    reader = csv.DictReader(after_file)
    for row in reader:
        selector = (os.path.basename(row['Filename']), int(row['ASIndex']))
        if selector not in after_times:
            after_times[selector] = []
        if selector not in after_sizes:
            after_sizes[selector] = []
        after_times[selector].append(float(row['BuildTime']))
        after_sizes[selector].append(float(row['CompactedSize']))

if not before_times:
    print("Could not open first measurements file!")
    exit(1)
if not after_times:
    print("Could not open second measurements file!")
    exit(1)

class Cell:
   def __init__(self, text, width, color):
      self.text = text
      self.width = width
      self.color = color

class Table:
   def __init__(self, padding=1):
      self.rows = []
      self.padding = ' ' * padding

   def add(self, text, width=1, row=-1, color=''):
      self.rows[row].append(Cell(self.padding + text + self.padding, width, color))
      return self

   def next_row(self):
      self.rows.append([])
      return self

   def hline(self):
      self.rows.append(None)
      return self

   def vline(self, row=-1):
      self.rows[row].append(Cell('|', 1, ''))
      return self

   def print(self):
      column_widths = []
      for row in self.rows:
         if not row:
            continue

         column_index = 0
         for cell in row:
            for i in range(len(column_widths), column_index + cell.width):
               column_widths.append(0)
            if cell.width == 1:
               column_widths[column_index] = max(column_widths[column_index], len(cell.text))
            column_index += cell.width

      for row in self.rows:
         if not row:
            continue

         column_index = 0
         for cell in row:
            last_width = len(cell.text)
            for i in range(cell.width - 1):
               last_width -= column_widths[column_index + i]
            column_widths[column_index + cell.width - 1] = max(column_widths[column_index + cell.width - 1], last_width)
            column_index += cell.width

      total_width = 0
      for width in column_widths:
         total_width += width

      canonicalized_rows = []

      for row_index in range(len(self.rows)):
         row = self.rows[row_index]
         if row:
            padded_row = []

            column_index = 0
            for cell in row:
               padded_cell = cell.color + cell.text + COLOR_RESET
               cell_character_width = 0
               for i in range(cell.width):
                  cell_character_width += column_widths[column_index]
                  column_index += 1

               if len(cell.text) < cell_character_width:
                  padded_cell = padded_cell + (' ' * (cell_character_width - len(cell.text)))

               padded_row.append(padded_cell)

            canonicalized_rows.append(padded_row)
            continue

         row = []

         neighbor_row_indices = []
         if row_index > 0:
            neighbor_row_indices.append(row_index - 1)
         if row_index + 1 < len(self.rows):
            neighbor_row_indices.append(row_index + 1)

         hline = ['-' for i in range(total_width)]

         for neighbor_row_index in neighbor_row_indices:
            char_index = 0
            column_index = 0
            for cell in self.rows[neighbor_row_index]:
               if cell.text == '|':
                  hline[char_index] = '+'
               for i in range(cell.width):
                  char_index += column_widths[column_index]
                  column_index += 1

         for column_index in range(len(column_widths)):
            has_vline = False
            if row_index > 0 and column_index < len(self.rows[row_index - 1]) and self.rows[row_index - 1][column_index].text == '|':
               has_vline = True
            elif row_index + 1 < len(self.rows) and column_index < len(self.rows[row_index + 1]) and self.rows[row_index + 1][column_index].text == '|':
               has_vline = True

            if has_vline:
               row.append('+')
               row.append('-' * (column_widths[column_index] - 1))
            else:
               row.append('-' * column_widths[column_index])

         canonicalized_rows.append(''.join(hline))

      for row in canonicalized_rows:
         for cell in row:
            print(cell, end='')
         print()

def compute_mean_stddev(measurements):
   if len(measurements) == 1:
      return (float(measurements[0]), 0.0)

   sum_measurements = 0.0
   for measurement in measurements:
       sum_measurements += float(measurement)
   mean = sum_measurements / len(measurements)
   sum_sq_deviations = 0.0
   for measurement in measurements:
       sum_sq_deviations += (measurement - mean)**2
   stddev = math.sqrt(sum_sq_deviations / len(measurements))
   return (mean, stddev)

results = Table()
results.hline()
results.next_row().vline().add('File name')
results.add('AS Index')
results.add('Time change')
results.add('Size change').vline()

header = dict()

results.hline()

def student_crit(ν):
    student_crits = [
        # val  # ν
        2.706, # 1
        4.303, # 2
        3.182, # 3
        2.776, # 4
        2.571, # 5
        2.447, # 6
        2.365, # 7
        2.306, # 8
        2.262, # 9
        2.228, # 10
        2.201, # 11
        2.179, # 12
        2.160, # 13
        2.145, # 14
        2.131, # 15
        2.120, # 16
        2.110, # 17
        2.101, # 18
        2.093, # 19
        2.086, # 20
        2.080, # 21
        2.074, # 22
        2.069, # 23
        2.064, # 24
        2.060, # 25
        2.056, # 26
        2.052, # 27
        2.048, # 28
        2.045, # 29
        2.042, # 30
        2.040, # 31
        2.037, # 32
        2.035, # 33
        2.032, # 34
        2.030, # 35
        2.028, # 36
        2.026, # 37
        2.024, # 38
        2.023, # 39
        2.021, # 40
        2.020, # 41
        2.018, # 42
        2.017, # 43
        2.015, # 44
        2.014, # 45
        2.013, # 46
        2.012, # 47
        2.011, # 48
        2.010, # 49
        2.009, # 50
        2.008, # 51
        2.007, # 52
        2.006, # 53
        2.005, # 54
        2.004, # 55
        2.003, # 56
        2.002, # 57
        2.002, # 58
        2.001, # 59
        2.000, # 60
        2.000, # 61
        1.999, # 62
        1.998, # 63
        1.998, # 64
        1.997, # 65
        1.997, # 66
        1.996, # 67
        1.995, # 68
        1.995, # 69
        1.994, # 70
        1.994, # 71
        1.993, # 72
        1.993, # 73
        1.993, # 74
        1.992, # 75
        1.992, # 76
        1.991, # 77
        1.991, # 78
        1.990, # 79
        1.990, # 80
        1.990, # 81
        1.989, # 82
        1.989, # 83
        1.989, # 84
        1.988, # 85
        1.988, # 86
        1.988, # 87
        1.987, # 88
        1.987, # 89
        1.987, # 90
        1.986, # 91
        1.986, # 92
        1.986, # 93
        1.986, # 94
        1.985, # 95
        1.985, # 96
        1.985, # 97
        1.984, # 98
        1.984, # 99
        1.984, # 100 and up
    ]
    if ν >= len(student_crits):
        return student_crits[len(student_crits) - 1]
    return student_crits[ν]

def print_change_uncertainty(before_measurements, after_measurements):
    before = compute_mean_stddev(before_measurements)
    after = compute_mean_stddev(after_measurements)

    # Variance in the measurement results comes from power management/submission timing
    # inconsistencies, which we can assume to be similar between two runs, so use the
    # pooled variance of the two runs to determine the uncertainty.
    dof = len(before_measurements) + len(after_measurements) - 2

    pooled_variance = (len(before_measurements) - 1) * before[1] + (len(after_measurements) - 1) * after[1];
    pooled_variance /= dof
    pooled_variance = math.sqrt(pooled_variance)

    change = (after[0] - before[0])
    uncertainty = student_crit(dof) * (pooled_variance * math.sqrt(1.0 / len(before_measurements) + 1.0 / len(after_measurements)))

    change_pct = 0
    if before[0] != 0:
        change_pct = (after[0] / before[0] - 1) * 100
    uncertainty_pct = 0
    if uncertainty != 0:
        uncertainty_pct = (uncertainty / before[0]) * 100
    color = ''
    if abs(change) > uncertainty:
        if change < 0:
            color = COLOR_GREEN
        else:
            color = COLOR_RED

    results.add(('+' if change >= 0 else '') + uncertain_number_to_str(change_pct, uncertainty_pct) + "%", color=color)

for accel_struct in before_times.keys():
   results.next_row().vline().add(accel_struct[0])
   results.add(str(accel_struct[1]))

   accel_struct_before_times = before_times[accel_struct]
   accel_struct_before_sizes = before_sizes[accel_struct]
   accel_struct_after_times = after_times[accel_struct]
   accel_struct_after_sizes = after_sizes[accel_struct]

   print_change_uncertainty(accel_struct_before_times, accel_struct_after_times)
   print_change_uncertainty(accel_struct_before_sizes, accel_struct_after_sizes)
   results.vline()

results.hline()

results.print()
