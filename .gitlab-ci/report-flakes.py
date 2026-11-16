#!/usr/bin/env python3
#
# Copyright © 2021 Google LLC
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

import argparse
import io
import re
import socket
import time


class Connection:
    def __init__(self, host: str, port: int, verbose: bool):
        self.verbose = verbose
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        self.s.setblocking(0)

    def send_line(self, line: str) -> None:
        if self.verbose:
            print(f"IRC: sending {line}")
        self.s.sendall((line + '\n').encode())

    def wait(self, secs: int) -> None:
        for i in range(secs):
            if self.verbose:
                while True:
                    try:
                        data = self.s.recv(1024)
                    except io.BlockingIOError:
                        break
                    if data == "":
                        break
                    for line in data.decode().split('\n'):
                        print(f"IRC: received {line}")
            time.sleep(1)

    def quit(self) -> None:
        self.send_line("QUIT")
        self.s.shutdown(socket.SHUT_WR)
        self.s.close()


def read_flakes(results: str) -> tuple[list[str], list[str]]:
    new_flakes = []
    known_flakes = []
    csv = re.compile("(.*),(.*),(.*)")
    for line in open(results, 'r').readlines():
        match = csv.match(line)
        if match.group(2) == "Flake":
            new_flakes.append(match.group(1))
        if match.group(2) == "KnownFlake":
            known_flakes.append(match.group(1))
    return new_flakes, known_flakes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str,
                        help='IRC server hostname', required=True)
    parser.add_argument('--port', type=int,
                        help='IRC server port', required=True)
    parser.add_argument('--results', type=str,
                        help='results.csv file from deqp-runner or piglit-runner', required=True)
    parser.add_argument('--channel', type=str,
                        help='Known flakes report channel', required=True)
    parser.add_argument('--url', type=str,
                        help='$CI_JOB_URL', required=True)
    parser.add_argument('--runner', type=str,
                        help='$CI_RUNNER_DESCRIPTION', required=True)
    parser.add_argument('--branch', type=str,
                        help='optional branch name')
    parser.add_argument('--branch-title', type=str,
                        help='optional branch title')
    parser.add_argument('--job', type=str,
                        help='$CI_JOB_ID', required=True)
    parser.add_argument('--verbose', "-v", action="store_true",
                        help='log IRC interactions')
    args = parser.parse_args()

    new_flakes, known_flakes = read_flakes(args.results)
    if not new_flakes and not known_flakes:
        if args.verbose:
            print("No flakes to report, exiting")
        exit(0)

    irc = Connection(args.host, args.port, args.verbose)

    # The nick needs to be something unique so that multiple runners
    # connecting at the same time don't race for one nick and get blocked.
    # freenode has a 16-char limit on nicks (9 is the IETF standard, but
    # various servers extend that).  So, trim off the common prefixes of the
    # runner name, and append the job ID so that software runners with more
    # than one concurrent job (think swrast) don't collide.  For freedreno,
    # that gives us a nick as long as db410c-N-JJJJJJJJ, and it'll be a while
    # before we make it to 9-digit jobs (we're at 7 so far).
    nick = args.runner
    nick = nick.replace('mesa-', '')
    nick += f'-{args.job}'
    irc.send_line(f"NICK {nick}")
    irc.send_line(f"USER {nick} unused unused: Gitlab CI Notifier")
    irc.wait(10)
    irc.send_line(f"JOIN {args.channel}")
    irc.wait(1)

    branchinfo = ""
    if args.branch:
        branchinfo = f" on branch {args.branch} ({args.branch_title})"
    irc.send_line(
        f"PRIVMSG {args.channel} :Flakes detected in job {args.url} on {args.runner}{branchinfo}:")

    for flake in new_flakes:
        irc.send_line(f"PRIVMSG {args.channel} :NEW {flake}")
    for flake in known_flakes:
        irc.send_line(f"PRIVMSG {args.channel} :{flake}")

    irc.send_line(
        f"PRIVMSG {args.channel} :See {args.url}/artifacts/browse/results/")

    irc.quit()


if __name__ == '__main__':
    main()
