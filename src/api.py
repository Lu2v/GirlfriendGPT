
"""Scaffolding to host a conversational chatbot on Steamship and Telegram."""

from typing import List, Optional, Type

import langchain
from langchain.agents import (
    Tool,
    initialize_agent,
    AgentType,
    AgentExecutor,
)
from langchain.memory import ConversationBufferWindowMemory
from pydantic import Field

from steamship.experimental.package_starters.telegram_bot import (
    TelegramBot,
    TelegramBotConfig,
)
from steamship.invocable import Config
from steamship_langchain.llms import OpenAIChat
from steamship_langchain.memory import ChatMessageHistory

from agent.base import LangChainAgentBot
from agent.tools.search import SearchTool
from agent.tools.selfie import SelfieTool
from agent.tools.speech import GenerateSpeechTool
from agent.tools.video_message import VideoMessageTool
from personalities import get_personality
from prompts import SUFFIX, FORMAT_INSTRUCTIONS, PERSONALITY_PROMPT


MODEL_NAME = "gpt-4"
TEMPERATURE = 0.7
VERBOSE = True

# Personalidad seleccionada para el bot.
PERSONALITY = "aura"

MEMORY_WINDOW_SIZE = 5

langchain.cache = None


class GirlFriendAIConfig(TelegramBotConfig):
    bot_token: str = Field(
        description=(
            "Telegram bot token. Create one with Telegram BotFather."
        )
    )

    elevenlabs_api_key: str = Field(
        default="",
        description="Optional ElevenLabs API key.",
    )

    elevenlabs_voice_id: str = Field(
        default="",
        description="Optional ElevenLabs voice ID.",
    )


class GirlfriendGPT(LangChainAgentBot, TelegramBot):
    """Conversational bot with a configurable personality."""

    config: GirlFriendAIConfig

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the configuration class."""
        return GirlFriendAIConfig

    def get_agent(self, chat_id: str) -> AgentExecutor:
        """Build the conversational agent using Aura's personality."""

        personality_prompt = get_personality(PERSONALITY)

        llm = OpenAIChat(
            client=self.client,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            verbose=VERBOSE,
        )

        tools = self.get_tools(chat_id=chat_id)
        memory = self.get_memory(chat_id)

        return initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": PERSONALITY_PROMPT.format(
                    personality=personality_prompt
                ),
                "suffix": SUFFIX,
                "format_instructions": FORMAT_INSTRUCTIONS,
            },
            verbose=VERBOSE,
            memory=memory,
        )

    def voice_tool(self) -> Optional[Tool]:
        """Create the optional text-to-speech tool."""

        api_key = self.config.elevenlabs_api_key
        voice_id = self.config.elevenlabs_voice_id

        if not api_key or not voice_id:
            return None

        return GenerateSpeechTool(
            client=self.client,
            voice_id=voice_id,
            elevenlabs_api_key=api_key,
        )

    def get_memory(self, chat_id: str):
        """Create conversation memory for this chat."""

        if self.context and self.context.invocable_instance_handle:
            instance_handle = self.context.invocable_instance_handle
        else:
            instance_handle = "local-instance-handle"

        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            chat_memory=ChatMessageHistory(
                client=self.client,
                key=f"history-{chat_id}-{instance_handle}",
            ),
            return_messages=True,
            k=MEMORY_WINDOW_SIZE,
        )

    def get_tools(self, chat_id: str) -> List[Tool]:
        """Return the tools available to the agent."""

        return [
            SearchTool(self.client),
            VideoMessageTool(self.client),
            SelfieTool(self.client),
        ]
