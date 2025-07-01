"""
This file is used to fix the notorious 'Received request before initialization was complete' error.
This issue is difficult to resolve due to its origin within the MCP library itself.
This patch simply overrides the function that throws the exception, allowing the service to run,
although it may come with unforeseen consequences.

Keep this patch in place until MCP addresses the issue.
For more information, please see https://github.com/modelcontextprotocol/python-sdk/issues/423
"""

from enum import Enum
import mcp.types as types
from mcp.shared.session import (
    RequestResponder,
)

from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from loguru import logger
import mcp


class InitializationState(Enum):
    NotInitialized = 1
    Initializing = 2
    Initialized = 3


async def _received_request(
    self, responder: RequestResponder[types.ClientRequest, types.ServerResult]
):
    match responder.request.root:
        case types.InitializeRequest(params=params):
            requested_version = params.protocolVersion
            self._initialization_state = InitializationState.Initializing
            self._client_params = params
            with responder:
                await responder.respond(
                    types.ServerResult(
                        types.InitializeResult(
                            protocolVersion=(
                                requested_version
                                if requested_version in SUPPORTED_PROTOCOL_VERSIONS
                                else types.LATEST_PROTOCOL_VERSION
                            ),
                            capabilities=self._init_options.capabilities,
                            serverInfo=types.Implementation(
                                name=self._init_options.server_name,
                                version=self._init_options.server_version,
                            ),
                            instructions=self._init_options.instructions,
                        )
                    )
                )
        case _:
            if self._initialization_state != InitializationState.Initialized:
                # raise RuntimeError("Received request before initialization was complete")
                # Override the error with a note to keep the service running.
                logger.trace("Received request before initialization was complete")


def apply_patch() -> None:
    """
    Apply the patch to the MCP library.
    """
    logger.info("Applying MCP initialization patch")
    mcp.server.session.ServerSession._received_request = _received_request
