import asyncio
import json
import logging
import os
import shutil
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

# MCP-Agent imports
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences

# Add the parent directory to the path for imports
# Import the GitHub issue creator
from github_issue_creator import create_github_issue

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP Slackbot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_app_token = os.getenv("SLACK_APP_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")

        # GitHub configuration
        self.github_default_owner = 'sessionlab'
        self.github_default_repo = 'sessionlab'

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the appropriate LLM API key based on the model.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If no API key is found for the selected model.
        """
        if "gpt" in self.llm_model.lower() and self.openai_api_key:
            return self.openai_api_key
        elif "llama" in self.llm_model.lower() and self.groq_api_key:
            return self.groq_api_key
        elif "claude" in self.llm_model.lower() and self.anthropic_api_key:
            return self.anthropic_api_key

        # Fallback to any available key
        if self.openai_api_key:
            return self.openai_api_key
        elif self.groq_api_key:
            return self.groq_api_key
        elif self.anthropic_api_key:
            return self.anthropic_api_key

        raise ValueError("No API key found for any LLM provider")


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: Dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Client for communicating with LLM APIs."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider
            model: Model identifier to use
        """
        self.api_key = api_key
        self.model = model
        self.timeout = 30.0  # 30 second timeout
        self.max_retries = 2

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: List of conversation messages

        Returns:
            Text response from the LLM
        """
        if self.model.startswith("gpt-") or self.model.startswith("ft:gpt-"):
            return await self._get_openai_response(messages)
        elif self.model.startswith("llama-"):
            return await self._get_groq_response(messages)
        elif self.model.startswith("claude-"):
            return await self._get_anthropic_response(messages)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    async def _get_openai_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff

    async def _get_groq_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the Groq API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff

    async def _get_anthropic_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append(
                    {"role": "assistant", "content": msg["content"]}
                )

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        if system_message:
            payload["system"] = system_message

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["content"][0]["text"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff


class SlackMCPBot:
    """Manages the Slack bot integration with MCP servers."""

    def __init__(
        self,
        slack_bot_token: str,
        slack_app_token: str,
        servers: List[Server],
        llm_client: LLMClient,
    ) -> None:
        self.app = AsyncApp(token=slack_bot_token)
        # Create a socket mode handler with the app token
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, slack_app_token)

        self.client = AsyncWebClient(token=slack_bot_token)
        self.servers = servers
        self.llm_client = llm_client
        self.conversations = {}  # Store conversation context per channel
        self.tools = []

        # Initialize MCP-Agent components
        self.mcp_app = MCPApp()
        self.summarizer_agent = None  # Will be initialized in start method

        # Set up event handlers
        self.app.event("app_mention")(self.handle_mention)
        self.app.message()(self.handle_message)
        self.app.event("app_home_opened")(self.handle_home_opened)

    async def initialize_servers(self) -> None:
        """Initialize all MCP servers and discover tools."""
        for server in self.servers:
            try:
                await server.initialize()
                server_tools = await server.list_tools()
                self.tools.extend(server_tools)
                logging.info(
                    f"Initialized server {server.name} with {len(server_tools)} tools"
                )
            except Exception as e:
                logging.error(f"Failed to initialize server {server.name}: {e}")

    async def initialize_bot_info(self) -> None:
        """Get the bot's ID and other info."""
        try:
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Failed to get bot info: {e}")
            self.bot_id = None

    async def handle_mention(self, event, say):
        """Handle mentions of the bot in channels."""
        # Check if this is a summarization request in a thread
        text = event.get("text", "").lower()
        if event.get("thread_ts") and "summarize" in text:
            logging.info('Handling thread summarization request')
            await self.handle_thread_summarization(event, say)
            return

        # Check if this is a GitHub issue creation request in a thread
        if event.get("thread_ts") and ("create bug" in text or "create github bug" in text):
            logging.info('Handling GitHub issue creation request')
            await self.handle_github_issue_creation(event, say)
            return

        # Otherwise, process as normal
        await self._process_message(event, say)

    async def handle_message(self, message, say):
        """Handle direct messages to the bot."""
        # Only process direct messages
        if message.get("channel_type") == "im" and not message.get("subtype"):
            await self._process_message(message, say)

    async def handle_thread_summarization(self, event, say):
        """Handle a thread summarization request."""
        logging.info('Receive summarization request')
        channel = event["channel"]
        thread_ts = event.get("thread_ts")

        if not thread_ts:
            await say(text="Please tag me within a thread to get a summary.")
            return

        # Add a reaction to show we're processing
        try:
            logging.info('Trying to add reaction')
            await self.client.reactions_add(
                channel=channel,
                timestamp=event["ts"],
                name="hourglass_flowing_sand"
            )
        except Exception as e:
            logging.warning(f"Could not add reaction: {e}")

        try:
            # Get thread messages
            messages = await self._get_thread_messages(channel, thread_ts)

            if not messages:
                await say(text="No messages found in this thread.", thread_ts=thread_ts)
                return

            # Format thread for summarization
            thread_text = self._format_thread_for_summarization(messages)

            # Generate summary
            summary = await self._summarize_thread(thread_text)

            # Send the summary
            await say(text=f"*Thread Summary:*\n\n{summary}", thread_ts=thread_ts)

            # Replace processing reaction with success reaction
            try:
                await self.client.reactions_remove(
                    channel=channel,
                    timestamp=event["ts"],
                    name="hourglass_flowing_sand"
                )
                await self.client.reactions_add(
                    channel=channel,
                    timestamp=event["ts"],
                    name="white_check_mark"
                )
            except Exception as e:
                logging.warning(f"Could not update reactions: {e}")

        except Exception as e:
            logging.error(f"Error summarizing thread: {e}", exc_info=True)
            await say(text=f"I encountered an error while summarizing this thread: {str(e)}", thread_ts=thread_ts)

            # Replace processing reaction with error reaction
            try:
                await self.client.reactions_remove(
                    channel=channel,
                    timestamp=event["ts"],
                    name="hourglass_flowing_sand"
                )
                await self.client.reactions_add(
                    channel=channel,
                    timestamp=event["ts"],
                    name="x"
                )
            except Exception:
                pass

    async def _get_thread_messages(self, channel, thread_ts):
        """Get all messages in a thread."""
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
                    if msg.get("user") == self.bot_id and "thread summary" in msg.get("text", "").lower():
                        continue
                    filtered_messages.append(msg)

                return filtered_messages
            return []
        except Exception as e:
            logging.error(f"Error fetching thread messages: {e}")
            return []

    def _format_thread_for_summarization(self, messages):
        """Format thread messages for summarization."""
        formatted_text = "Here is the conversation thread:\n\n"

        for i, msg in enumerate(messages):
            # Get user info
            user_id = msg.get("user", "unknown")
            username = f"<@{user_id}>"  # Use Slack user mention format

            # Format the message
            text = msg.get("text", "").strip()
            formatted_text += f"Message {i+1} - {username}:\n{text}\n\n"

        return formatted_text

    def _format_thread_for_issue(self, messages):
        """Format thread messages for GitHub issue creation."""
        formatted_text = "## Slack Thread Content\n\n"

        for i, msg in enumerate(messages):
            # Get user info
            user_id = msg.get("user", "unknown")
            username = f"@{user_id}"  # Use @ for GitHub markdown

            # Format the message
            text = msg.get("text", "").strip()
            formatted_text += f"**Message {i+1} - {username}:**\n{text}\n\n"

        return formatted_text

    async def _summarize_thread(self, thread_text):
        """Generate a summary using mcp-agent."""
        try:
            # Use the existing summarizer_agent
            if not self.summarizer_agent:
                logging.error("Summarizer agent not initialized")
                raise ValueError("Summarizer agent not initialized")

            async with self.summarizer_agent:
                # Choose LLM based on model name
                if "gpt" in self.llm_client.model.lower():
                    llm = await self.summarizer_agent.attach_llm(OpenAIAugmentedLLM)
                else:
                    llm = await self.summarizer_agent.attach_llm(AnthropicAugmentedLLM)

                # Configure request parameters
                request_params = RequestParams(
                    modelPreferences=ModelPreferences(
                        costPriority=0.3,
                        speedPriority=0.2,
                        intelligencePriority=0.5
                    )
                )

                # Generate summary
                prompt = f"""
                Please analyze and summarize the following Slack conversation thread.

                Focus on:
                1. The main topics discussed
                2. Any key decisions or conclusions reached
                3. Any action items or next steps mentioned
                4. The overall sentiment and tone

                Provide a concise but comprehensive summary.

                {thread_text}
                """

                summary = await llm.generate_str(
                    message=prompt,
                    request_params=request_params
                )

                return summary
        except Exception as e:
            logging.error(f"Error in _summarize_thread: {e}", exc_info=True)
            raise

    async def _extract_issue_details(self, event, thread_text):
        """Extract GitHub issue details from the request and thread."""
        try:
            # Use the GitHub issue creator agent
            if not self.github_issue_agent:
                logging.error("GitHub issue agent not initialized")
                raise ValueError("GitHub issue agent not initialized")

            async with self.github_issue_agent:
                # Choose LLM based on model name
                if "gpt" in self.llm_client.model.lower():
                    llm = await self.github_issue_agent.attach_llm(OpenAIAugmentedLLM)
                else:
                    llm = await self.github_issue_agent.attach_llm(AnthropicAugmentedLLM)

                # Configure request parameters
                request_params = RequestParams(
                    modelPreferences=ModelPreferences(
                        costPriority=0.3,
                        speedPriority=0.2,
                        intelligencePriority=0.5
                    )
                )

                # Extract repository information from the message
                text = event.get("text", "").lower()

                # Default repository information from configuration
                config = Configuration()
                owner = config.github_default_owner
                repo = config.github_default_repo

                # Try to extract repository information from the message
                # Format could be "create issue for owner/repo"
                if "for" in text:
                    parts = text.split("for", 1)[1].strip().split()
                    if parts and "/" in parts[0]:
                        repo_parts = parts[0].split("/")
                        if len(repo_parts) == 2:
                            owner = repo_parts[0].strip()
                            repo = repo_parts[1].strip()

                # Generate a title and body for the issue
                prompt = f"""
                Please analyze the following Slack conversation thread and extract information for a GitHub issue.

                Create:
                1. A clear, concise title for the issue (one line)
                2. A detailed description for the issue body

                Make sure the title summarizes the main point, and the body provides enough context.
                Do not include any labels or assignees in your response.
                Format your answer as JSON with "title" and "body" fields.

                {thread_text}
                """

                response = await llm.generate_str(
                    message=prompt,
                    request_params=request_params
                )

                # Parse the JSON response
                try:
                    import json
                    details = json.loads(response)

                    # Make sure we have the required fields
                    if "title" not in details or "body" not in details:
                        raise ValueError("Missing required fields in issue details")

                    # Add the repository information
                    details["owner"] = owner
                    details["repo"] = repo

                    return details
                except json.JSONDecodeError:
                    # If the response isn't valid JSON, try to extract title and body manually
                    title_match = None
                    body = ""

                    lines = response.split("\n")
                    for i, line in enumerate(lines):
                        if "title" in line.lower() and ":" in line:
                            title_match = line.split(":", 1)[1].strip()
                            body = "\n".join(lines[i+1:]).strip()
                            break

                    if not title_match:
                        # If we can't find a clear title, use the first line
                        title_match = lines[0].strip()
                        body = "\n".join(lines[1:]).strip()

                    return {
                        "title": title_match,
                        "body": body,
                        "owner": owner,
                        "repo": repo
                    }

        except Exception as e:
            logging.error(f"Error in _extract_issue_details: {e}", exc_info=True)
            raise

    async def handle_github_issue_creation(self, event, say):
        """Handle creating a GitHub issue from a thread."""
        channel = event["channel"]
        thread_ts = event.get("thread_ts")

        if not thread_ts:
            await say(text="Please tag me within a thread to create a GitHub issue.")
            return

        # Add a reaction to show we're processing
        try:
            logging.info('Adding processing reaction for GitHub issue creation')
            await self.client.reactions_add(
                channel=channel,
                timestamp=event["ts"],
                name="hourglass_flowing_sand"
            )
        except Exception as e:
            logging.warning(f"Could not add reaction: {e}")

        try:
            # Get thread messages
            messages = await self._get_thread_messages(channel, thread_ts)

            if not messages:
                await say(text="No messages found in this thread.", thread_ts=thread_ts)
                return

            # Format thread for issue creation
            thread_text = self._format_thread_for_issue(messages)

            # Extract repository information from the message
            text = event.get("text", "").lower()

            # Default repository information from configuration
            config = Configuration()
            owner = config.github_default_owner
            repo = config.github_default_repo

            # Try to extract repository information from the message
            # Format could be "create issue for owner/repo"
            if "for" in text:
                parts = text.split("for", 1)[1].strip().split()
                if parts and "/" in parts[0]:
                    repo_parts = parts[0].split("/")
                    if len(repo_parts) == 2:
                        owner = repo_parts[0].strip()
                        repo = repo_parts[1].strip()

            # Let the user know we're working on it
            await say(
                text=f"Creating GitHub issue for {owner}/{repo}...",
                thread_ts=thread_ts
            )

            # Determine which LLM to use
            use_anthropic = "claude" in self.llm_client.model.lower()

            # Use the dedicated agent to create the issue
            result = await create_github_issue(
                thread_content=thread_text,
                owner=owner,
                repo=repo,
                use_anthropic=use_anthropic
            )

            if result["success"]:
                # Issue created successfully
                if result.get("issue_url"):
                    await say(
                        text=f"✅ GitHub issue created successfully: {result['issue_url']}",
                        thread_ts=thread_ts
                    )
                else:
                    # We have a success but no URL, include the response
                    await say(
                        text=f"✅ GitHub issue created successfully.\n\n{result['response']}",
                        thread_ts=thread_ts
                    )

                # Replace processing reaction with success reaction
                try:
                    await self.client.reactions_remove(
                        channel=channel,
                        timestamp=event["ts"],
                        name="hourglass_flowing_sand"
                    )
                    await self.client.reactions_add(
                        channel=channel,
                        timestamp=event["ts"],
                        name="white_check_mark"
                    )
                except Exception as e:
                    logging.warning(f"Could not update reactions: {e}")

                    #     return
                    # except Exception as e:
                    #     logging.error(f"Error creating GitHub issue: {e}", exc_info=True)
                    #     await say(
                    #         text=f"❌ Failed to create GitHub issue: {str(e)}",
                    #         thread_ts=thread_ts
                    #     )

                    #     # Replace processing reaction with error reaction
                    #     try:
                    #         await self.client.reactions_remove(
                    #             channel=channel,
                    #             timestamp=event["ts"],
                    #             name="hourglass_flowing_sand"
                    #         )
                    #         await self.client.reactions_add(
                    #             channel=channel,
                    #             timestamp=event["ts"],
                    #             name="x"
                    #         )
                    #     except Exception:
                    #         pass

                    #     return

            # If we get here, we didn't find the GitHub server

        except Exception as e:
            logging.error(f"Error creating GitHub issue: {e}", exc_info=True)
            await say(text=f"I encountered an error while creating the GitHub issue: {str(e)}", thread_ts=thread_ts)

            # Replace processing reaction with error reaction
            try:
                await self.client.reactions_remove(
                    channel=channel,
                    timestamp=event["ts"],
                    name="hourglass_flowing_sand"
                )
                await self.client.reactions_add(
                    channel=channel,
                    timestamp=event["ts"],
                    name="x"
                )
            except Exception:
                pass

    async def handle_home_opened(self, event, client):
        """Handle when a user opens the App Home tab."""
        user_id = event["user"]

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Welcome to MCP Assistant!"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "I'm an AI assistant with access to tools and resources "
                        "through the Model Context Protocol."
                    ),
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Available Tools:*"},
            },
        ]

        # Add tools
        for tool in self.tools:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• *{tool.name}*: {tool.description}",
                    },
                }
            )

        # Add usage section
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*How to Use:*\n• Send me a direct message\n"
                        "• Mention me in a channel with @MCP Assistant\n"
                        "• *Thread Summarization:* Tag me in a thread with the word 'summarize' to get a summary of the conversation\n"
                        "• *GitHub Issue Creation:* Tag me in a thread with 'create issue' to create a GitHub issue from the thread"
                    ),
                },
            }
        )

        # Add thread summarization section with more details
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*Thread Summarization Feature:*\n"
                        "I can summarize Slack threads for you! Just:\n"
                        "1. Reply to any message in a thread\n"
                        "2. Tag me and include the word 'summarize'\n"
                        "3. I'll read all messages in the thread and create a concise summary\n"
                        "4. Great for catching up on long discussions or documenting decisions"
                    ),
                },
            }
        )

        # Add GitHub issue creation section with details
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*GitHub Issue Creation Feature:*\n"
                        "I can create GitHub issues from Slack threads! Just:\n"
                        "1. Reply to any message in a thread\n"
                        "2. Tag me and include 'create issue' or 'create github issue'\n"
                        "3. Optionally specify repository with 'create issue for owner/repo'\n"
                        "4. I'll create an issue with details extracted from the thread\n"
                        "5. I'll post back the link to the created issue"
                    ),
                },
            }
        )

        try:
            await client.views_publish(
                user_id=user_id, view={"type": "home", "blocks": blocks}
            )
        except Exception as e:
            logging.error(f"Error publishing home view: {e}")

    async def _process_message(self, event, say):
        """Process incoming messages and generate responses."""
        channel = event["channel"]
        user_id = event.get("user")

        # Skip messages from the bot itself
        if user_id == getattr(self, "bot_id", None):
            return

        # Get text and remove bot mention if present
        text = event.get("text", "")
        if hasattr(self, "bot_id") and self.bot_id:
            text = text.replace(f"<@{self.bot_id}>", "").strip()

        thread_ts = event.get("thread_ts", event.get("ts"))

        # Get or create conversation context
        if channel not in self.conversations:
            self.conversations[channel] = {"messages": []}

        try:
            # Create system message with tool descriptions
            tools_text = "\n".join([tool.format_for_llm() for tool in self.tools])
            system_message = {
                "role": "system",
                "content": (
                    f"""You are a helpful assistant with access to the following tools:

{tools_text}

When you need to use a tool, you MUST format your response exactly like this:
[TOOL] tool_name
{{"param1": "value1", "param2": "value2"}}

Make sure to include both the tool name AND the JSON arguments.
Never leave out the JSON arguments.

After receiving tool results, interpret them for the user in a helpful way.
"""
                ),
            }

            # Add user message to history
            self.conversations[channel]["messages"].append(
                {"role": "user", "content": text}
            )

            # Set up messages for LLM
            messages = [system_message]

            # Add conversation history (last 5 messages)
            if "messages" in self.conversations[channel]:
                messages.extend(self.conversations[channel]["messages"][-5:])

            # Get LLM response
            response = await self.llm_client.get_response(messages)

            # Process tool calls in the response
            if "[TOOL]" in response:
                response = await self._process_tool_call(response, channel)

            # Add assistant response to conversation history
            self.conversations[channel]["messages"].append(
                {"role": "assistant", "content": response}
            )

            # Send the response to the user
            await say(text=response, channel=channel, thread_ts=thread_ts)

        except Exception as e:
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            logging.error(f"Error processing message: {e}", exc_info=True)
            await say(text=error_message, channel=channel, thread_ts=thread_ts)

    async def _process_tool_call(self, response: str, channel: str) -> str:
        """Process a tool call from the LLM response."""
        try:
            # Extract tool name and arguments
            tool_parts = response.split("[TOOL]")[1].strip().split("\n", 1)
            tool_name = tool_parts[0].strip()

            # Handle incomplete tool calls
            if len(tool_parts) < 2:
                return (
                    f"I tried to use the tool '{tool_name}', but the request "
                    f"was incomplete. Here's my response without the tool:"
                    f"\n\n{response.split('[TOOL]')[0]}"
                )

            # Parse JSON arguments
            try:
                args_text = tool_parts[1].strip()
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                return (
                    f"I tried to use the tool '{tool_name}', but the arguments "
                    f"were not properly formatted. Here's my response without "
                    f"the tool:\n\n{response.split('[TOOL]')[0]}"
                )

            # Find the appropriate server for this tool
            for server in self.servers:
                server_tools = [tool.name for tool in await server.list_tools()]
                if tool_name in server_tools:
                    # Execute the tool
                    tool_result = await server.execute_tool(tool_name, arguments)

                    # Add tool result to conversation history
                    tool_result_msg = f"Tool result for {tool_name}:\n{tool_result}"
                    self.conversations[channel]["messages"].append(
                        {"role": "system", "content": tool_result_msg}
                    )

                    try:
                        # Get interpretation from LLM
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful assistant. You've just "
                                    "used a tool and received results. Interpret "
                                    "these results for the user in a clear, "
                                    "helpful way."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"I used the tool {tool_name} with arguments "
                                    f"{args_text} and got this result:\n\n"
                                    f"{tool_result}\n\n"
                                    f"Please interpret this result for me."
                                ),
                            },
                        ]

                        interpretation = await self.llm_client.get_response(messages)
                        return interpretation
                    except Exception as e:
                        logging.error(
                            f"Error getting tool result interpretation: {e}",
                            exc_info=True,
                        )
                        # Fallback to basic formatting
                        if isinstance(tool_result, dict):
                            result_text = json.dumps(tool_result, indent=2)
                        else:
                            result_text = str(tool_result)
                        return (
                            f"I used the {tool_name} tool and got these results:"
                            f"\n\n```\n{result_text}\n```"
                        )

            # No server had the tool
            return (
                f"I tried to use the tool '{tool_name}', but it's not available. "
                f"Here's my response without the tool:\n\n{response.split('[TOOL]')[0]}"
            )

        except Exception as e:
            logging.error(f"Error executing tool: {e}", exc_info=True)
            return (
                f"I tried to use a tool, but encountered an error: {str(e)}\n\n"
                f"Here's my response without the tool:\n\n{response.split('[TOOL]')[0]}"
            )

    async def start(self) -> None:
        """Start the Slack bot."""
        await self.initialize_servers()
        await self.initialize_bot_info()

        # Initialize the summarizer agent
        self.summarizer_agent = Agent(
            name="thread_summarizer",
            instruction="You analyze Slack conversation threads and provide concise summaries.",
            server_names=[]  # No server needs for basic summarization
        )
        logging.info("Thread summarizer agent initialized")

        # Note: We don't need to initialize the GitHub issue creator agent here
        # since we're using the dedicated agent module instead

        # Start the socket mode handler
        logging.info("Starting Slack bot...")
        asyncio.create_task(self.socket_mode_handler.start_async())
        logging.info("Slack bot started and waiting for messages")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "socket_mode_handler"):
                await self.socket_mode_handler.close_async()
            logging.info("Slack socket mode handler closed")
        except Exception as e:
            logging.error(f"Error closing socket mode handler: {e}")

        # Clean up servers
        for server in self.servers:
            try:
                await server.cleanup()
                logging.info(f"Server {server.name} cleaned up")
            except Exception as e:
                logging.error(f"Error during cleanup of server {server.name}: {e}")


async def main() -> None:
    """Initialize and run the Slack bot."""
    config = Configuration()

    if not config.slack_bot_token or not config.slack_app_token:
        raise ValueError(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in environment variables"
        )

    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]

    llm_client = LLMClient(config.llm_api_key, config.llm_model)

    slack_bot = SlackMCPBot(
        config.slack_bot_token, config.slack_app_token, servers, llm_client
    )

    try:
        await slack_bot.start()
        # Keep the main task alive until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await slack_bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
