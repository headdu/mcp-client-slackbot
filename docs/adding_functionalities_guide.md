# Adding Functionalities to the MCP-Agent Slack Bot

This guide explains how to extend the MCP-Agent Slack Bot with new capabilities. Whether you want to add new event handlers, integrate additional MCP servers, or implement complex workflows, this guide provides step-by-step instructions with practical examples.

## Table of Contents

1. [Introduction](#introduction)
2. [Adding New Event Handlers](#adding-new-event-handlers)
3. [Creating Specialized Agents](#creating-specialized-agents)
4. [Integrating Additional MCP Servers](#integrating-additional-mcp-servers)
5. [Implementing Workflow Patterns](#implementing-workflow-patterns)
6. [Custom LLM Integration](#custom-llm-integration)
7. [Extension Best Practices](#extension-best-practices)
8. [Example Extensions](#example-extensions)

## Introduction

The MCP-Agent Slack Bot is designed to be extensible, allowing you to add new features without major refactoring. This guide covers the main extension points and provides practical examples for each.

Key extension points include:
- Slack event handlers
- Agent capabilities
- MCP server connections
- Workflow patterns
- LLM providers

Before adding new functionalities, make sure you understand the existing [architecture and implementation](implementation_guide.md).

## Adding New Event Handlers

Slack bots respond to various events such as messages, reactions, file uploads, and more. The current implementation handles app mentions and direct messages, but you can easily add handlers for other events.

### Step 1: Register a New Event Handler

First, add the new event type to the `__init__` method of the `SlackMCPBot` class:

```python
def __init__(self, ...):
    # Existing initialization...

    # Add a new event handler
    self.app.event("reaction_added")(self.handle_reaction_added)

    # Existing event handlers...
    self.app.event("app_mention")(self.handle_mention)
    self.app.message()(self.handle_message)
```

### Step 2: Implement the Handler Method

Add a method to handle the event:

```python
async def handle_reaction_added(self, event, say):
    """Handle reactions added to messages."""
    reaction = event.get("reaction")
    item = event.get("item", {})
    channel = item.get("channel")
    ts = item.get("ts")

    # Only process specific reactions (e.g., "eyes")
    if reaction == "eyes" and channel and ts:
        try:
            # Fetch the message that was reacted to
            response = await self.client.conversations_history(
                channel=channel,
                latest=ts,
                limit=1,
                inclusive=True
            )

            if response and response.get("messages"):
                message = response["messages"][0]
                text = message.get("text", "")

                # Process the message
                await say(
                    text=f"I noticed you were interested in this message: {text}",
                    thread_ts=ts
                )

        except Exception as e:
            logging.error(f"Error handling reaction: {e}")
```

### Step 3: Request the Required Scopes

For the bot to receive certain events, you need to add the appropriate scopes in your Slack app configuration:

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Select your app
3. Navigate to "OAuth & Permissions"
4. Add the required scopes (e.g., `reactions:read` for reaction events)
5. Reinstall your app to the workspace

### Example: File Upload Handler

Here's an example of a handler for file uploads:

```python
def __init__(self, ...):
    # Existing initialization...
    self.app.event("file_shared")(self.handle_file_shared)

async def handle_file_shared(self, event, say):
    """Process shared files."""
    file_id = event.get("file_id")
    channel = event.get("channel_id")

    if not file_id or not channel:
        return

    try:
        # Get file information
        file_info = await self.client.files_info(file=file_id)
        file = file_info.get("file", {})

        # Only process certain file types
        if file.get("filetype") in ["csv", "xlsx", "pdf"]:
            await say(
                text=f"I noticed you shared a {file.get('filetype')} file. Would you like me to analyze it?",
                channel=channel
            )
    except Exception as e:
        logging.error(f"Error handling file: {e}")
```

## Creating Specialized Agents

You can create specialized agents for specific tasks using the mcp-agent framework.

### Step 1: Define the Agent Purpose

First, define what the agent should do. For example, you might create:
- A research agent that fetches and summarizes information
- A data analysis agent that processes data files
- A creative writing agent that generates content

### Step 2: Create the Agent

Add a method to initialize the agent:

```python
def initialize_agents(self):
    """Initialize specialized agents."""
    # Existing initialization...

    # Create a research agent
    self.research_agent = Agent(
        name="research_assistant",
        instruction="""You are a research assistant that can search the web and summarize
        information. Your goal is to provide accurate, concise summaries of topics when asked.""",
        server_names=["fetch"]  # This agent uses the fetch server
    )

    logging.info("Research agent initialized")
```

### Step 3: Implement Agent Usage

Add a method that uses the agent:

```python
async def handle_research_request(self, event, say):
    """Handle research requests."""
    text = event.get("text", "").lower()
    topic = text.replace("research", "").strip()
    thread_ts = event.get("thread_ts", event.get("ts"))
    channel = event["channel"]

    if not topic:
        await say(text="Please specify a research topic.", thread_ts=thread_ts)
        return

    # Add processing reaction
    try:
        await self.client.reactions_add(
            channel=channel,
            timestamp=event["ts"],
            name="hourglass_flowing_sand"
        )
    except Exception as e:
        logging.warning(f"Could not add reaction: {e}")

    try:
        async with self.research_agent:
            # Attach appropriate LLM
            llm = await self.research_agent.attach_llm(AnthropicAugmentedLLM)

            # Generate research
            prompt = f"""
            Please research the following topic and provide a comprehensive summary:

            Topic: {topic}

            Include key facts, important figures, and relevant context.
            """

            research = await llm.generate_str(message=prompt)

            # Send the research
            await say(text=f"*Research on {topic}:*\n\n{research}", thread_ts=thread_ts)

            # Update reaction
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
        logging.error(f"Error generating research: {e}")
        await say(text=f"I encountered an error while researching this topic: {str(e)}", thread_ts=thread_ts)

        # Update reaction
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
```

### Step 4: Update the Router

Modify your message handling to route to the new function:

```python
async def handle_mention(self, event, say):
    """Handle mentions of the bot in channels."""
    text = event.get("text", "").lower()

    # Check if this is a summarization request
    if event.get("thread_ts") and "summarize" in text:
        await self.handle_thread_summarization(event, say)
        return

    # Check if this is a research request
    if "research" in text:
        await self.handle_research_request(event, say)
        return

    # Otherwise, process as normal
    await self._process_message(event, say)
```

## Integrating Additional MCP Servers

MCP servers provide tools that agents can use. Adding a new server extends the capabilities of your agents.

### Step 1: Add Server Configuration

Update your `servers_config.json` file with the new server:

```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "description": "Fetch web content from a URL"
    },
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "./test.db"],
      "description": "SQLite database access for storing and querying data"
    },
    "google_search": {
      "command": "npx",
      "args": ["-y", "mcp-server-google-search"],
      "description": "Google search functionality"
    }
  }
}
```

### Step 2: Add Environment Variables

If the server requires API keys, add them to your `.env` file:

```
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-custom-search-engine-id
```

### Step 3: Update Agent Configuration

When creating an agent, specify the new server in the `server_names` list:

```python
self.advanced_research_agent = Agent(
    name="advanced_researcher",
    instruction="""You are an advanced research assistant that can search the web using Google
    to find specific information. Your goal is to provide accurate, up-to-date information
    about any topic.""",
    server_names=["fetch", "google_search"]  # This agent uses both servers
)
```

### Step 4: Create Helper Methods

For complex server interactions, create helper methods:

```python
async def perform_google_search(self, query, num_results=5):
    """Perform a Google search using the google_search MCP server."""
    for server in self.servers:
        if server.name == "google_search":
            try:
                search_result = await server.execute_tool(
                    "google_search",
                    {
                        "query": query,
                        "num_results": num_results
                    }
                )
                return search_result
            except Exception as e:
                logging.error(f"Error executing Google search: {e}")
                raise

    raise ValueError("Google search server not found")
```

## Implementing Workflow Patterns

The mcp-agent library provides several workflow patterns that can be used to build complex agents.

### Router Pattern

The Router pattern directs requests to the appropriate handler based on their content:

```python
from mcp_agent.workflows.router import LLMRouter

async def initialize_router(self):
    """Initialize a router for directing requests."""
    self.router = LLMRouter(
        categories=[
            {
                "name": "summarization",
                "description": "Summarizes text content like threads or articles"
            },
            {
                "name": "research",
                "description": "Searches for and summarizes information on a topic"
            },
            {
                "name": "data_analysis",
                "description": "Analyzes data files and provides insights"
            },
            {
                "name": "creative",
                "description": "Generates creative content like stories or marketing copy"
            }
        ],
        top_k=1  # Return the top match
    )

async def route_request(self, text):
    """Route a request to the appropriate handler."""
    result = await self.router.route(text)
    category = result[0]["name"] if result else None

    if category == "summarization":
        return "summarization"
    elif category == "research":
        return "research"
    elif category == "data_analysis":
        return "data_analysis"
    elif category == "creative":
        return "creative"
    else:
        return "general"
```

### Parallel Pattern

The Parallel pattern runs multiple agents concurrently and combines their results:

```python
from mcp_agent.workflows.parallel import Parallel

async def analyze_document(self, document_text):
    """Analyze a document from multiple perspectives."""

    # Create specialized agents
    content_agent = Agent(
        name="content_analyzer",
        instruction="Analyze the content of the document, extracting key themes and topics.",
        server_names=[]
    )

    structure_agent = Agent(
        name="structure_analyzer",
        instruction="Analyze the structure of the document, focusing on organization and flow.",
        server_names=[]
    )

    style_agent = Agent(
        name="style_analyzer",
        instruction="Analyze the style and tone of the document.",
        server_names=[]
    )

    # Create parallel workflow
    parallel = Parallel(
        subtasks=[content_agent, structure_agent, style_agent],
        aggregator=self._combine_analysis_results
    )

    # Run analysis
    result = await parallel.generate_str(document_text)
    return result

def _combine_analysis_results(self, results):
    """Combine the results from multiple analysis agents."""
    combined = "# Document Analysis\n\n"
    combined += "## Content Analysis\n" + results[0] + "\n\n"
    combined += "## Structure Analysis\n" + results[1] + "\n\n"
    combined += "## Style Analysis\n" + results[2] + "\n\n"
    return combined
```

### Evaluator-Optimizer Pattern

The Evaluator-Optimizer pattern improves outputs through iterative refinement:

```python
from mcp_agent.workflows.evaluator_optimizer import EvaluatorOptimizer

async def generate_optimized_response(self, prompt):
    """Generate a response that is iteratively improved."""

    # Define evaluation criteria
    evaluation_prompt = """
    Evaluate the response based on the following criteria:
    1. Accuracy - Is the information correct?
    2. Completeness - Does it address all aspects of the query?
    3. Clarity - Is it clear and easy to understand?
    4. Conciseness - Is it appropriately concise?

    Score each criterion from 1-5 and provide overall feedback.
    """

    # Create evaluator-optimizer workflow
    eo = EvaluatorOptimizer(
        optimizer_system_prompt="You are an assistant that generates high-quality responses to user queries.",
        evaluator_system_prompt="You are an evaluator that assesses the quality of responses.",
        evaluation_prompt=evaluation_prompt,
        min_score=4.0,  # Minimum average score required
        max_iterations=3  # Maximum number of improvement iterations
    )

    # Generate optimized response
    result = await eo.generate_str(prompt)
    return result
```

## Custom LLM Integration

You can integrate additional LLM providers beyond the default ones.

### Step 1: Configure API Keys

Add the API key to your `.env` file or `mcp_agent.secrets.yaml`:

```yaml
# mcp_agent.secrets.yaml
mistral:
  api_key: "your-mistral-api-key"
```

### Step 2: Update Configuration

Add the new provider to your `mcp_agent.config.yaml`:

```yaml
mistral:
  default_model: "mistral-large-latest"
```

### Step 3: Create an LLM Adapter

Create a new file `mistral_llm.py` with the adapter implementation:

```python
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams
from typing import Dict, List, Optional, Any
import os
import httpx

class MistralAugmentedLLM(AugmentedLLM):
    """AugmentedLLM implementation for Mistral AI."""

    def __init__(
        self,
        model: str = "mistral-large-latest",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")

        if not self.api_key:
            raise ValueError("Mistral API key is required")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        request_params: Optional[RequestParams] = None,
    ) -> Dict[str, Any]:
        """Generate a response using Mistral AI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        # Apply request parameters if provided
        if request_params:
            if request_params.temperature is not None:
                payload["temperature"] = request_params.temperature
            if request_params.max_tokens is not None:
                payload["max_tokens"] = request_params.max_tokens

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                json=payload,
                headers=headers
            )

            if response.status_code != 200:
                raise Exception(f"Mistral API error: {response.text}")

            result = response.json()
            return {
                "content": result["choices"][0]["message"]["content"],
                "finish_reason": result["choices"][0]["finish_reason"]
            }
```

### Step 4: Use the New LLM

Update your agent to use the new LLM:

```python
from mistral_llm import MistralAugmentedLLM

async def use_mistral_llm(self, prompt):
    """Use Mistral LLM for a specific task."""

    # Create an agent
    agent = Agent(
        name="mistral_agent",
        instruction="You provide insightful responses using the Mistral language model.",
        server_names=[]
    )

    async with agent:
        # Attach Mistral LLM
        llm = MistralAugmentedLLM(
            model="mistral-large-latest",
            temperature=0.5,
            max_tokens=1500
        )

        # Generate response
        response = await llm.generate_str(message=prompt)
        return response
```

## Extension Best Practices

Follow these best practices when extending the bot:

### Modular Design

- Keep extensions separate from core functionality
- Use clear interfaces between components
- Minimize dependencies between extensions

```python
# Good: Modular approach
async def handle_special_request(self, event, say):
    """Handle a specialized request type."""
    # Implementation...

# Bad: Mixing multiple responsibilities
async def handle_mention(self, event, say):
    """Handle mentions with embedded specialized handling."""
    text = event.get("text", "").lower()

    if "research" in text:
        # Research implementation directly here...
    elif "summarize" in text:
        # Summarization implementation directly here...
    # ...and so on
```

### Error Handling

- Always wrap external API calls in try-except blocks
- Provide user-friendly error messages
- Log detailed error information for debugging
- Include cleanup code to remove reactions and release resources

```python
# Good error handling
try:
    result = await some_operation()
    # Process result...
except SomeSpecificError as e:
    logging.error(f"Specific error: {e}")
    await say("I encountered a specific problem...")
except Exception as e:
    logging.error(f"Unexpected error: {e}", exc_info=True)
    await say("I'm sorry, something unexpected happened...")
finally:
    # Cleanup code (remove reactions, close connections, etc.)
```

### Configuration Management

- Make extensions configurable through environment variables or config files
- Use sensible defaults
- Validate configuration at startup
- Document configuration options

```python
# Good configuration approach
def __init__(self, ...):
    # Load configuration with defaults
    self.research_model = os.getenv("RESEARCH_MODEL", "claude-3-7-sonnet-20250219")
    self.research_max_results = int(os.getenv("RESEARCH_MAX_RESULTS", "5"))
    self.research_temperature = float(os.getenv("RESEARCH_TEMPERATURE", "0.3"))

    # Validate configuration
    if self.research_max_results <= 0:
        raise ValueError("RESEARCH_MAX_RESULTS must be positive")
    if not (0.0 <= self.research_temperature <= 1.0):
        raise ValueError("RESEARCH_TEMPERATURE must be between 0.0 and 1.0")
```

### Testing

- Write unit tests for new functionality
- Create integration tests for complex workflows
- Test error cases and edge conditions
- Use mocks for external dependencies

```python
# Example test for a new handler
async def test_handle_research_request(self):
    # Mock dependencies
    mock_client = MagicMock()
    mock_client.conversations_replies.return_value = {...}

    mock_say = AsyncMock()

    # Create test event
    event = {
        "type": "app_mention",
        "text": "<@U123> research artificial intelligence",
        "channel": "C123",
        "ts": "1234567890.123456"
    }

    # Create bot instance with mocks
    bot = SlackMCPBot(...)
    bot.client = mock_client

    # Run the handler
    await bot.handle_research_request(event, mock_say)

    # Assertions
    mock_say.assert_called_once()
    assert "artificial intelligence" in mock_say.call_args[1]["text"]
```

## Example Extensions

Here are some example extensions to inspire your own development:

### Meeting Notes Summarizer

```python
async def handle_meeting_notes(self, event, say):
    """Summarize meeting notes from a thread."""
    thread_ts = event.get("thread_ts")
    channel = event["channel"]

    if not thread_ts:
        await say("Please use this command in a thread containing meeting notes.")
        return

    # Get thread messages
    messages = await self._get_thread_messages(channel, thread_ts)

    # Format as meeting notes
    notes_text = "# Meeting Notes\n\n"
    for msg in messages:
        user_id = msg.get("user", "unknown")
        text = msg.get("text", "").strip()
        notes_text += f"- <@{user_id}>: {text}\n\n"

    # Create specialized agent for meeting notes
    meeting_agent = Agent(
        name="meeting_summarizer",
        instruction="""You summarize meeting notes into a structured format with:
        1. Attendees
        2. Key Discussion Points
        3. Decisions Made
        4. Action Items (with owners)
        5. Next Steps""",
        server_names=[]
    )

    async with meeting_agent:
        # Attach LLM
        llm = await meeting_agent.attach_llm(AnthropicAugmentedLLM)

        # Generate summary
        prompt = f"""
        Please summarize these meeting notes into a structured format.

        {notes_text}
        """

        meeting_summary = await llm.generate_str(message=prompt)

        # Send the summary
        await say(text=f"*Meeting Summary:*\n\n{meeting_summary}", thread_ts=thread_ts)
```

### Data Analysis Agent

```python
async def handle_data_analysis(self, event, say):
    """Analyze data files shared in Slack."""
    file_id = event.get("file", {}).get("id")
    channel = event["channel"]
    thread_ts = event.get("thread_ts", event.get("ts"))

    if not file_id:
        await say("Please share a data file to analyze.", thread_ts=thread_ts)
        return

    # Download the file
    file_info = await self.client.files_info(file=file_id)
    file_url = file_info.get("file", {}).get("url_private")

    if not file_url:
        await say("Unable to access the file.", thread_ts=thread_ts)
        return

    # Download using authorized request
    headers = {"Authorization": f"Bearer {self.slack_bot_token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(file_url, headers=headers)
        file_content = response.content

    # Create a data analysis agent
    data_agent = Agent(
        name="data_analyst",
        instruction="You analyze data files and provide insights.",
        server_names=["pandas"]  # Assuming you have a pandas MCP server
    )

    async with data_agent:
        # Attach LLM
        llm = await data_agent.attach_llm(OpenAIAugmentedLLM)

        # Send file to agent
        result = await data_agent.session.call_tool(
            "analyze_data",
            {
                "file_content": base64.b64encode(file_content).decode("utf-8"),
                "file_type": file_info.get("file", {}).get("filetype", "csv")
            }
        )

        # Get insights from the analysis
        prompt = f"""
        Based on the data analysis results, provide key insights and recommendations:

        {result}
        """

        insights = await llm.generate_str(message=prompt)

        # Send the insights
        await say(text=f"*Data Analysis Insights:*\n\n{insights}", thread_ts=thread_ts)
```

This guide provides an overview of how to extend the MCP-Agent Slack Bot with new functionalities. By following these patterns and best practices, you can create powerful, specialized capabilities that enhance the bot's utility.
