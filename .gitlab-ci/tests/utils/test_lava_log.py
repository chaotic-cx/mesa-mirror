#!/usr/bin/env python3
#
# Copyright (C) 2022 Collabora Limited
# Author: Guilherme Gallo <guilherme.gallo@collabora.com>
#
# SPDX-License-Identifier: MIT

from datetime import UTC, datetime, timedelta

import pytest

from lava.exceptions import MesaCIKnownIssueException, MesaCITimeoutError
from lava.utils import (
    GitlabSection,
    LogFollower,
    LogSectionType,
    fix_lava_gitlab_section_log,
    hide_sensitive_data,
)
from lava.utils.constants import (
    KNOWN_ISSUE_R8152_MAX_CONSECUTIVE_COUNTER,
    A6XX_GPU_RECOVERY_WATCH_PERIOD_MIN,
    A6XX_GPU_RECOVERY_FAILURE_MESSAGE,
    A6XX_GPU_RECOVERY_FAILURE_MAX_COUNT,
)
from lava.utils.lava_log_hints import LAVALogHints
from ..lava.helpers import (
    create_lava_yaml_msg,
    does_not_raise,
    lava_yaml,
    mock_lava_signal,
    yaml_dump,
)

GITLAB_SECTION_SCENARIOS = {
    "start collapsed": (
        "start",
        True,
        f"\x1b[0Ksection_start:mock_date:my_first_section[collapsed=true]\r\x1b[0K"
        f"{GitlabSection.colour}my_header\x1b[0m",
    ),
    "start non_collapsed": (
        "start",
        False,
        f"\x1b[0Ksection_start:mock_date:my_first_section\r\x1b[0K"
        f"{GitlabSection.colour}my_header\x1b[0m",
    ),
    "end collapsed": (
        "end",
        True,
        "\x1b[0Ksection_end:mock_date:my_first_section\r\x1b[0K",
    ),
    "end non_collapsed": (
        "end",
        False,
        "\x1b[0Ksection_end:mock_date:my_first_section\r\x1b[0K",
    ),
}


@pytest.mark.parametrize(
    "method, collapsed, expectation",
    GITLAB_SECTION_SCENARIOS.values(),
    ids=GITLAB_SECTION_SCENARIOS.keys(),
)
def test_gitlab_section(method, collapsed, expectation):
    gs = GitlabSection(
        id="my_first_section",
        header="my_header",
        type=LogSectionType.TEST_CASE,
        start_collapsed=collapsed,
    )
    gs.get_timestamp = lambda mock_date: "mock_date"
    gs.start()
    result = getattr(gs, method)()
    assert result == expectation


def test_gl_sections():
    lines = [
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "debug",
            "msg": "Received signal: <STARTRUN> 0_setup-ssh-server 10145749_1.3.2.3.1",
        },
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "debug",
            "msg": "Received signal: <STARTRUN> 0_mesa 5971831_1.3.2.3.1",
        },
        # Redundant log message which triggers the same Gitlab Section, it
        # should be ignored, unless the id is different
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "target",
            "msg": "[    7.778836] <LAVA_SIGNAL_STARTRUN 0_mesa 5971831_1.3.2.3.1>",
        },
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "debug",
            "msg": "Received signal: <STARTTC> mesa-ci_iris-kbl-traces",
        },
        # Another redundant log message
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "target",
            "msg": "[   16.997829] <LAVA_SIGNAL_STARTTC mesa-ci_iris-kbl-traces>",
        },
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "target",
            "msg": "<LAVA_SIGNAL_ENDTC mesa-ci_iris-kbl-traces>",
        },
    ]
    lf = LogFollower()
    with lf:
        for line in lines:
            lf.manage_gl_sections(line)
        parsed_lines = lf.flush()

    section_types = [s.type for s in lf.section_history]

    assert "section_start" in parsed_lines[0]
    assert "collapsed=true" in parsed_lines[0]
    assert "section_end" in parsed_lines[1]
    assert "section_start" in parsed_lines[2]
    assert "collapsed=true" in parsed_lines[2]
    assert "section_end" in parsed_lines[3]
    assert "section_start" in parsed_lines[4]
    assert "collapsed=true" in parsed_lines[4]
    assert section_types == [
        # LogSectionType.LAVA_BOOT,  True, if LogFollower started with Boot section
        LogSectionType.TEST_SUITE,
        LogSectionType.TEST_CASE,
        LogSectionType.LAVA_POST_PROCESSING,
    ]


def test_log_follower_flush():
    lines = [
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "debug",
            "msg": "Received signal: <STARTTC> mesa-ci_iris-kbl-traces",
        },
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "target",
            "msg": "<LAVA_SIGNAL_ENDTC mesa-ci_iris-kbl-traces>",
        },
    ]
    lf = LogFollower()
    lf.feed(lines)
    parsed_lines = lf.flush()
    empty = lf.flush()
    lf.feed(lines)
    repeated_parsed_lines = lf.flush()

    assert parsed_lines
    assert not empty
    assert repeated_parsed_lines


SENSITIVE_DATA_SCENARIOS = {
    "no sensitive data tagged": (
        ["bla  bla", "mytoken: asdkfjsde1341=="],
        ["bla  bla", "mytoken: asdkfjsde1341=="],
        ["HIDEME"],
    ),
    "sensitive data tagged": (
        ["bla  bla", "mytoken: asdkfjsde1341== # HIDEME"],
        ["bla  bla"],
        ["HIDEME"],
    ),
    "sensitive data tagged with custom word": (
        ["bla  bla", "mytoken: asdkfjsde1341== # DELETETHISLINE", "third line # NOTANYMORE"],
        ["bla  bla", "third line # NOTANYMORE"],
        ["DELETETHISLINE", "NOTANYMORE"],
    ),
}


@pytest.mark.parametrize(
    "input, expectation, tags",
    SENSITIVE_DATA_SCENARIOS.values(),
    ids=SENSITIVE_DATA_SCENARIOS.keys(),
)
def test_hide_sensitive_data(input, expectation, tags):
    yaml_data = yaml_dump(input)
    yaml_result = hide_sensitive_data(yaml_data, *tags)
    result = lava_yaml.load(yaml_result)

    assert result == expectation


GITLAB_SECTION_SPLIT_SCENARIOS = {
    "Split section_start at target level": (
        "\x1b[0Ksection_start:1668454947:test_post_process[collapsed=true]\r\x1b[0K"
        "post-processing test results",
        (
            "\x1b[0Ksection_start:1668454947:test_post_process[collapsed=true]",
            "\x1b[0Kpost-processing test results",
        ),
    ),
    "Split section_end at target level": (
        "\x1b[0Ksection_end:1666309222:test_post_process\r\x1b[0K",
        ("\x1b[0Ksection_end:1666309222:test_post_process", "\x1b[0K"),
    ),
    "Second line is not split from the first": (
        ("\x1b[0Ksection_end:1666309222:test_post_process", "Any message"),
        ("\x1b[0Ksection_end:1666309222:test_post_process", "Any message"),
    ),
}


@pytest.mark.parametrize(
    "expected_message, messages",
    GITLAB_SECTION_SPLIT_SCENARIOS.values(),
    ids=GITLAB_SECTION_SPLIT_SCENARIOS.keys(),
)
def test_fix_lava_gitlab_section_log(expected_message, messages):
    fixed_messages = []
    gen = fix_lava_gitlab_section_log()
    next(gen)

    for message in messages:
        lava_log = create_lava_yaml_msg(msg=message, lvl="target")
        if recovered_line := gen.send(lava_log):
            fixed_messages.append((recovered_line, lava_log["msg"]))
        fixed_messages.append(lava_log["msg"])

    assert expected_message in fixed_messages


@pytest.mark.parametrize(
    "expected_message, messages",
    GITLAB_SECTION_SPLIT_SCENARIOS.values(),
    ids=GITLAB_SECTION_SPLIT_SCENARIOS.keys(),
)
def test_lava_gitlab_section_log_collabora(expected_message, messages, monkeypatch):
    """Check if LogFollower does not change the message if we are running in Collabora farm."""
    monkeypatch.setenv("RUNNER_TAG", "mesa-ci-x86_64-lava-test")
    lf = LogFollower()
    for message in messages:
        lf.feed([create_lava_yaml_msg(msg=message)])
    new_messages = lf.flush()
    new_messages = tuple(new_messages) if len(new_messages) > 1 else new_messages[0]
    assert new_messages == expected_message


CARRIAGE_RETURN_SCENARIOS = {
    "Carriage return at the end of the previous line": (
        (
            "\x1b[0Ksection_start:1677609903:test_setup[collapsed=true]\r\x1b[0K\x1b[0;36m[303:44] "
            "deqp: preparing test setup\x1b[0m",
        ),
        (
            "\x1b[0Ksection_start:1677609903:test_setup[collapsed=true]\r",
            "\x1b[0K\x1b[0;36m[303:44] deqp: preparing test setup\x1b[0m\r\n",
        ),
    ),
    "Newline at the end of the line": (
        ("\x1b[0K\x1b[0;36m[303:44] deqp: preparing test setup\x1b[0m", "log"),
        ("\x1b[0K\x1b[0;36m[303:44] deqp: preparing test setup\x1b[0m\r\n", "log"),
    ),
}


@pytest.mark.parametrize(
    "expected_message, messages",
    CARRIAGE_RETURN_SCENARIOS.values(),
    ids=CARRIAGE_RETURN_SCENARIOS.keys(),
)
def test_lava_log_merge_carriage_return_lines(expected_message, messages):
    lf = LogFollower()
    for message in messages:
        lf.feed([create_lava_yaml_msg(msg=message)])
    new_messages = tuple(lf.flush())
    assert new_messages == expected_message


WATCHDOG_SCENARIOS = {
    "1 second before timeout": ({"seconds": -1}, does_not_raise()),
    "1 second after timeout": ({"seconds": 1}, pytest.raises(MesaCITimeoutError)),
}


@pytest.mark.parametrize(
    "timedelta_kwargs, exception",
    WATCHDOG_SCENARIOS.values(),
    ids=WATCHDOG_SCENARIOS.keys(),
)
def test_log_follower_watchdog(frozen_time, timedelta_kwargs, exception):
    lines = [
        {
            "dt": datetime.now(tz=UTC),
            "lvl": "debug",
            "msg": "Received signal: <STARTTC> mesa-ci_iris-kbl-traces",
        },
    ]
    td = {LogSectionType.TEST_CASE: timedelta(minutes=1)}
    lf = LogFollower(timeout_durations=td)
    lf.feed(lines)
    frozen_time.tick(
        lf.timeout_durations[LogSectionType.TEST_CASE] + timedelta(**timedelta_kwargs)
    )
    lines = [create_lava_yaml_msg()]
    with exception:
        lf.feed(lines)


GITLAB_SECTION_ID_SCENARIOS = [
    ("a-good_name", "a-good_name"),
    ("spaces are not welcome", "spaces-are-not-welcome"),
    ("abc:amd64 1/3", "abc-amd64-1-3"),
]


@pytest.mark.parametrize("case_name, expected_id", GITLAB_SECTION_ID_SCENARIOS)
def test_gitlab_section_id(case_name, expected_id):
    gl = GitlabSection(
        id=case_name, header=case_name, type=LogSectionType.LAVA_POST_PROCESSING
    )

    assert gl.id == expected_id


def a618_network_issue_logs(level: str = "target") -> list:
    net_error = create_lava_yaml_msg(
            msg="[ 1733.599402] r8152 2-1.3:1.0 eth0: Tx status -71", lvl=level)

    nfs_error = create_lava_yaml_msg(
            msg="[ 1733.604506] nfs: server 192.168.201.1 not responding, still trying",
            lvl=level,
        )

    return [
        *(KNOWN_ISSUE_R8152_MAX_CONSECUTIVE_COUNTER*[net_error]),
        nfs_error
    ]


TEST_PHASE_LAVA_SIGNAL = mock_lava_signal(LogSectionType.TEST_CASE)
A618_NET_ISSUE_BOOT = a618_network_issue_logs(level="feedback")
A618_NET_ISSUE_TEST = [TEST_PHASE_LAVA_SIGNAL, *a618_network_issue_logs(level="target")]


A618_NETWORK_ISSUE_SCENARIOS = {
    "Fail - R8152 kmsg during boot phase": (
        A618_NET_ISSUE_BOOT,
        pytest.raises(MesaCIKnownIssueException),
    ),
    "Fail - R8152 kmsg during test phase": (
        A618_NET_ISSUE_TEST,
        pytest.raises(MesaCIKnownIssueException),
    ),
    "Pass - Partial (1) R8152 kmsg during test phase": (
        A618_NET_ISSUE_TEST[:1],
        does_not_raise(),
    ),
    "Pass - Partial (2) R8152 kmsg during test phase": (
        A618_NET_ISSUE_TEST[:2],
        does_not_raise(),
    ),
    "Pass - Partial (3) subsequent R8152 kmsg during test phase": (
        [
            TEST_PHASE_LAVA_SIGNAL,
            A618_NET_ISSUE_TEST[1],
            A618_NET_ISSUE_TEST[1],
        ],
        does_not_raise(),
    ),
    "Pass - Partial (4) subsequent nfs kmsg during test phase": (
        [
            TEST_PHASE_LAVA_SIGNAL,
            A618_NET_ISSUE_TEST[-1],
            A618_NET_ISSUE_TEST[-1],
        ],
        does_not_raise(),
    ),
}


@pytest.mark.parametrize(
    "messages, expectation",
    A618_NETWORK_ISSUE_SCENARIOS.values(),
    ids=A618_NETWORK_ISSUE_SCENARIOS.keys(),
)
def test_detect_failure(messages, expectation):
    boot_section = GitlabSection(
        id="dut_boot",
        header="Booting hardware device",
        type=LogSectionType.LAVA_BOOT,
        start_collapsed=True,
    )
    boot_section.start()
    lf = LogFollower(starting_section=boot_section)
    with expectation:
        lf.feed(messages)


def test_detect_a6xx_gpu_recovery_failure(frozen_time):
    log_follower = LogFollower()
    lava_log_hints = LAVALogHints(log_follower=log_follower)
    failure_message = {
        "dt": datetime.now(tz=UTC).isoformat(),
        "msg": A6XX_GPU_RECOVERY_FAILURE_MESSAGE[0],
        "lvl": "feedback",
    }
    with pytest.raises(MesaCIKnownIssueException):
        for _ in range(A6XX_GPU_RECOVERY_FAILURE_MAX_COUNT):
            lava_log_hints.detect_a6xx_gpu_recovery_failure(failure_message)
            # Simulate the passage of time within the watch period
            frozen_time.tick(1)
            failure_message["dt"] = datetime.now(tz=UTC).isoformat()


def test_detect_a6xx_gpu_recovery_success(frozen_time):
    log_follower = LogFollower()
    lava_log_hints = LAVALogHints(log_follower=log_follower)
    failure_message = {
        "dt": datetime.now(tz=UTC).isoformat(),
        "msg": A6XX_GPU_RECOVERY_FAILURE_MESSAGE[0],
        "lvl": "feedback",
    }
    # Simulate sending a tolerable number of failure messages
    for _ in range(A6XX_GPU_RECOVERY_FAILURE_MAX_COUNT - 1):
        lava_log_hints.detect_a6xx_gpu_recovery_failure(failure_message)
        frozen_time.tick(1)
        failure_message["dt"] = datetime.now(tz=UTC).isoformat()

    # Simulate the passage of time outside of the watch period
    frozen_time.tick(60 * A6XX_GPU_RECOVERY_WATCH_PERIOD_MIN + 1)
    failure_message = {
        "dt": datetime.now(tz=UTC).isoformat(),
        "msg": A6XX_GPU_RECOVERY_FAILURE_MESSAGE[1],
        "lvl": "feedback",
    }
    with does_not_raise():
        lava_log_hints.detect_a6xx_gpu_recovery_failure(failure_message)
    assert lava_log_hints.a6xx_gpu_first_fail_time is None, (
        "a6xx_gpu_first_fail_time is not None"
    )
    assert lava_log_hints.a6xx_gpu_recovery_fail_counter == 0, (
        "a6xx_gpu_recovery_fail_counter is not 0"
    )


@pytest.mark.parametrize(
    "start_offset",
    [
        timedelta(hours=0),
        timedelta(hours=1),
    ],
    ids=["equal timestamps", "negative delta"],
)
def test_gitlab_section_relative_time_clamping(start_offset):
    """Test that delta time is clamped to zero if start_time <= timestamp_relative_to."""
    now = datetime.now(tz=UTC)
    timestamp_relative_to = now + start_offset
    gs = GitlabSection(
        id="clamp_section",
        header=f"clamp_section header {start_offset}",
        type=LogSectionType.TEST_CASE,
        timestamp_relative_to=timestamp_relative_to,
    )
    gs.start()
    output = gs.print_start_section()
    assert "[00:00]" in output, f"Expected clamped relative time, got: {output}"


@pytest.mark.parametrize(
    "delta_seconds,expected_seconds",
    [
        (-5, 0),  # Negative delta should be clamped to 0
        (0, 0),  # Zero delta should remain 0
        (5, 5),  # Positive delta should remain unchanged
    ],
    ids=["negative delta", "zero delta", "positive delta"],
)
def test_gitlab_section_delta_time(frozen_time, delta_seconds, expected_seconds):
    """Test that delta_time() properly clamps negative deltas to zero."""
    gs = GitlabSection(
        id="delta_section",
        header=f"delta_section header {delta_seconds}",
        type=LogSectionType.TEST_CASE,
    )

    with gs:
        frozen_time.tick(delta_seconds)

    # Test internal _delta_time() returns exact delta
    internal_delta = gs._delta_time()
    assert internal_delta == timedelta(seconds=delta_seconds), (
        f"_delta_time() returned {internal_delta}, expected {timedelta(seconds=delta_seconds)}"
    )

    # Test public delta_time() returns clamped delta
    clamped_delta = gs.delta_time()
    assert clamped_delta == timedelta(seconds=expected_seconds), (
        f"delta_time() returned {clamped_delta}, expected {timedelta(seconds=expected_seconds)}"
    )
