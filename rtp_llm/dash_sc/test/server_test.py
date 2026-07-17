from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.dash_sc.server import dash_sc_grpc_server_channel_options


class _FakeDashScGrpcConfig:
    def __init__(self) -> None:
        self._server_config: dict[str, int] = {}

    def from_json(self, json_str: str) -> None:
        self._server_config = json.loads(json_str)["server_config"]

    def get_server_config(self) -> dict[str, int]:
        return self._server_config


class DashScGrpcServerChannelOptionsTest(TestCase):
    def test_default_config_applies_receive_message_limit(self) -> None:
        fake_ops = ModuleType("rtp_llm.ops")
        fake_ops.DashScGrpcConfig = _FakeDashScGrpcConfig

        with patch.dict(sys.modules, {"rtp_llm.ops": fake_ops}):
            options = dict(dash_sc_grpc_server_channel_options(None))

        self.assertEqual(
            options["grpc.max_receive_message_length"],
            64 * 1024 * 1024,
        )


if __name__ == "__main__":
    main()
