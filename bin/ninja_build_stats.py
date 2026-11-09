#!/usr/bin/env python3

import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ninja_log',
                        type=pathlib.Path,
                        help='Path to the build/.ninja_log file')
    parser.add_argument('-n',
                        type=int,
                        default=25,
                        help='Number of objects to show')
    args = parser.parse_args()

    stats = []
    for line in args.ninja_log.read_text().splitlines():
        if line.startswith('# ninja log'):
            continue
        begin, end, _, obj, _ = line.split()
        stats.append((obj, int(end) - int(begin)))

    print('Longest build times (in seconds)')
    for stat in sorted(stats, key=lambda i: i[1], reverse=True)[:args.n]:
        print(f'{stat[1] / 1000.0:5.1f} {stat[0]}')


if __name__ == '__main__':
    main()
