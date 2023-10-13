"""Callback Handler use a logger to log information among chains."""
from logging import Logger
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.schema.messages import BaseMessage


class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler for logging."""
    def __init__(self, logger: Logger):
        self.logger = logger

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.logger.info("Chat model started running.")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.logger.info("LLM finished running.")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors."""
        self.logger.error(f"LLM error: {error}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        self.logger.info("Chain started running.")
        self.logger.info(f"Inputs: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.logger.info("Chain finished running.")
        self.logger.info(f"Outputs: {outputs}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors."""
        self.logger.error(f"Chain error: {error}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        self.logger.info("Tool started running.")
        self.logger.info(f"Input: {input_str}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.logger.info("Tool finished running.")
        self.logger.info(f"Output: {output}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when tool errors."""
        self.logger.error(f"Tool error: {error}")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        self.logger.info(f"Text: {text}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        self.logger.info("Agent finished.")
