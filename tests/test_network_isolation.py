"""Proves the suite runs with network sockets disabled (pytest-socket)."""

import socket

import pytest
from pytest_socket import SocketBlockedError


@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6], ids=["inet", "inet6"])
def test_inet_sockets_are_blocked(family: socket.AddressFamily) -> None:
    with (
        pytest.warns(UserWarning, match=r"socket\.socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket(family, socket.SOCK_STREAM)
