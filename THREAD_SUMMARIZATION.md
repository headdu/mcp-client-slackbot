# Thread Summarization Feature

This document provides detailed information about the thread summarization feature in the MCP Slackbot.

## Overview

The thread summarization feature allows users to get an AI-generated summary of a Slack thread by simply tagging the bot with the keyword "summarize" in a thread reply. This feature leverages the mcp-agent library to provide advanced summarization capabilities.

## How It Works

1. A user tags the bot in a thread with a message containing "summarize"
2. The bot reacts with a ⏳ emoji to show it's processing
3. The bot fetches all messages in the thread
4. Messages are formatted and sent to an LLM through mcp-agent
5. The LLM generates a concise summary focusing on key points, decisions, and action items
6. The bot posts the summary as a reply in the thread
7. The bot adds a ✅ emoji to indicate successful completion

## Setup

### Prerequisites

- A functional Slack bot with the necessary permissions (see main README.md)
- The mcp-agent library installed (`pip install mcp-agent`)
- API keys for either Anthropic Claude or OpenAI GPT

### Configuration

The thread summarization feature uses the mcp-agent configuration files:

1. **mcp_agent.config.yaml**: Main configuration
   - Configures the LLM provider, temperature, and max tokens
   - Sets up any required MCP servers

2. **mcp_agent.secrets.yaml**: API keys (created from the example file)
   - Contains API keys for Anthropic, OpenAI, or Groq

## Usage Examples

Here are some ways to use the thread summarization feature:

### Basic Summarization

```
@MCP Assistant please summarize this thread
```

### Specific Requests

```
@MCP Assistant can you summarize this discussion and highlight the key decisions?
```

```
@MCP Assistant summarize this thread and extract action items
```

## Customization

You can customize the feature by modifying:

- The prompt template in the `_summarize_thread` method
- The LLM parameters (temperature, max_tokens) for different summary styles
- The reaction emojis used to indicate processing and completion

## Troubleshooting

If the feature isn't working as expected:

1. Check that your LLM API keys are correctly set up
2. Verify the bot has the necessary permissions to access thread history
3. Look for any errors in the bot logs
4. Ensure the thread has enough substantive content to summarize

## Limitations

- The bot cannot summarize very large threads due to token limits
- Some message formatting (like code blocks or complex Slack markdown) may be simplified
- The quality of summaries depends on the underlying LLM model
