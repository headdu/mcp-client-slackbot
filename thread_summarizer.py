#!/usr/bin/env python3
"""
Thread Summarizer Agent using MCP-Agent

This script implements a Slack bot that listens for mentions in threads,
fetches all messages from the thread, and uses MCP-agent to summarize them.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any

# Slack related imports
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient

# MCP-Agent related imports
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ThreadSummarizerBot:
    """Slack bot that summarizes thread messages using MCP-agent."""

    def __init__(self):
        """Initialize the bot with necessary tokens and configurations."""
        # Get Slack tokens from environment variables
        self.slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
        self.slack_app_token = os.environ.get("SLACK_APP_TOKEN")

        # Check if tokens are provided
        if not self.slack_bot_token or not self.slack_app_token:
            raise ValueError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in environment variables")

        # Initialize Slack app
        self.app = AsyncApp(token=self.slack_bot_token)
        self.client = AsyncWebClient(token=self.slack_bot_token)

        # Create socket mode handler
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, self.slack_app_token)

        # Initialize MCP app
        self.mcp_app = MCPApp()

        # Set up event handlers
        self.app.event("app_mention")(self.handle_mention)

        # Bot ID will be set later
        self.bot_id = None

    async def initialize(self):
        """Initialize the bot and get its ID."""
        try:
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logger.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logger.error(f"Failed to get bot info: {e}")
            raise

    async def handle_mention(self, event, say):
        """
        Handle mentions of the bot in threads.

        Args:
            event: The Slack event data
            say: Function to send messages to Slack
        """
        try:
            # Only respond to mentions in threads
            if "thread_ts" in event:
                # Start typing indicator to show the bot is processing
                await self.client.reactions_add(
                    channel=event["channel"],
                    timestamp=event["ts"],
                    name="hourglass_flowing_sand"
                )

                # Get all thread messages
                thread_messages = await self._get_thread_messages(
                    channel=event["channel"],
                    thread_ts=event["thread_ts"]
                )

                if thread_messages:
                    # Create the prompt for summarization
                    thread_text = self._format_thread_for_summarization(thread_messages)

                    # Generate summary using MCP-agent
                    summary = await self._generate_summary(thread_text)

                    # Send the summary back to the thread
                    await say(text=summary, thread_ts=event["thread_ts"])

                    # Remove the processing indicator
                    await self.client.reactions_remove(
                        channel=event["channel"],
                        timestamp=event["ts"],
                        name="hourglass_flowing_sand"
                    )

                    # Add a checkmark to indicate completion
                    await self.client.reactions_add(
                        channel=event["channel"],
                        timestamp=event["ts"],
                        name="white_check_mark"
                    )
                else:
                    await say(
                        text="I couldn't find any messages in this thread to summarize.",
                        thread_ts=event["thread_ts"]
                    )
            else:
                # If not in a thread, provide instructions
                await say(
                    text="Please tag me within a thread to get a summary of the conversation.",
                    thread_ts=event.get("ts")
                )

        except Exception as e:
            logger.error(f"Error handling mention: {e}", exc_info=True)

            # Remove the processing indicator if it exists
            try:
                await self.client.reactions_remove(
                    channel=event["channel"],
                    timestamp=event["ts"],
                    name="hourglass_flowing_sand"
                )
            except:
                pass

            # Add an error indicator
            try:
                await self.client.reactions_add(
                    channel=event["channel"],
                    timestamp=event["ts"],
                    name="x"
                )
            except:
                pass

            # Notify the user
            await say(
                text=f"I encountered an error while summarizing this thread: {str(e)}",
                thread_ts=event.get("thread_ts", event.get("ts"))
            )

    async def _get_thread_messages(self, channel: str, thread_ts: str) -> List[Dict[str, Any]]:
        """
        Get all messages from a thread.

        Args:
            channel: The Slack channel ID
            thread_ts: The thread timestamp

        Returns:
            List of thread messages
        """
        try:
            response = await self.client.conversations_replies(
                channel=channel,
                ts=thread_ts
            )

            if response and "messages" in response:
                # Filter out any bot messages that are summaries to avoid recursion
                filtered_messages = []
                for msg in response["messages"]:
                    # Skip our own summary messages
                    if msg.get("user") == self.bot_id and "summary of the conversation" in msg.get("text", "").lower():
                        continue
                    filtered_messages.append(msg)

                return filtered_messages
            return []
        except Exception as e:
            logger.error(f"Error getting thread messages: {e}", exc_info=True)
            return []

    def _format_thread_for_summarization(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format thread messages for summarization.

        Args:
            messages: List of thread messages

        Returns:
            Formatted text for the summarization agent
        """
        formatted_text = "Here is the conversation thread:\n\n"

        for i, msg in enumerate(messages):
            # Get user info (in a real implementation, you might want to cache this)
            user_id = msg.get("user", "unknown")
            username = f"<@{user_id}>"  # Use Slack user mention format

            # Format the message
            text = msg.get("text", "").strip()
            formatted_text += f"Message {i+1} - {username}:\n{text}\n\n"

        return formatted_text

    async def _generate_summary(self, thread_text: str) -> str:
        """
        Generate a summary of the thread using MCP-agent.

        Args:
            thread_text: Formatted thread text

        Returns:
            Generated summary
        """
        try:
            # Create an agent for summarization
            summarizer_agent = Agent(
                name="thread_summarizer",
                instruction="You analyze Slack conversation threads and provide concise summaries.",
                server_names=[]  # No server needs for basic summarization
            )

            async with summarizer_agent:
                # Get API key config from environment
                if os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
                    # Use OpenAI if only OpenAI key is available
                    llm = await summarizer_agent.attach_llm(OpenAIAugmentedLLM)
                else:
                    # Default to Anthropic (Claude) if available or if both are available
                    llm = await summarizer_agent.attach_llm(AnthropicAugmentedLLM)

                # Generate the summary with more advanced configuration
                prompt = f"""
                Please analyze and summarize the following Slack conversation thread.

                Focus on:
                1. The main topics discussed
                2. Any key decisions or conclusions reached
                3. Any action items or next steps mentioned
                4. The overall tone and sentiment of the conversation

                Provide a concise but comprehensive summary that someone who hasn't read the thread could understand.

                {thread_text}
                """

                # Configure request parameters for better results
                request_params = RequestParams(
                    modelPreferences=ModelPreferences(
                        costPriority=0.3,         # Balance cost with quality
                        speedPriority=0.2,        # Not super urgent
                        intelligencePriority=0.5  # Higher priority on intelligence for better summaries
                    )
                )

                summary = await llm.generate_str(
                    message=prompt,
                    request_params=request_params
                )

                # Format the response
                formatted_summary = f"*Thread Summary:*\n\n{summary}"

                return formatted_summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            raise

    async def start(self):
        """Start the Slack bot."""
        await self.initialize()
        await self.socket_mode_handler.start_async()
        logger.info("Thread Summarizer Bot is running!")

    async def stop(self):
        """Stop the Slack bot."""
        await self.socket_mode_handler.close_async()
        logger.info("Thread Summarizer Bot has been stopped.")


async def main():
    """Main function to run the bot."""
    bot = ThreadSummarizerBot()

    try:
        await bot.start()
        # Keep the bot running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
