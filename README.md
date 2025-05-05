# MCP Client Slackbot

A Slack bot that integrates with Model Context Protocol (MCP) servers, including thread summarization features powered by mcp-agent.

## Features

- Integration with various MCP servers (fetch, sqlite, git, github, etc.)
- Support for multiple LLM providers (OpenAI, Anthropic, Groq)
- Handles direct messages and mentions
- Executes tools through MCP servers
- **Thread Summarization**: Summarizes Slack conversation threads on demand

## Project Structure

The repository contains two main implementations:

1. **mcp_simple_slackbot**: A standalone Slack bot that integrates with MCP servers
2. **thread_summarizer.py**: A dedicated thread summarization bot using the mcp-agent framework

## Thread Summarization Feature

The bot can summarize Slack conversation threads! This feature is powered by the mcp-agent library, which provides advanced agent workflows for complex tasks.

### How to use thread summarization:

1. In any Slack thread, mention the bot with the word "summarize"
   ```
   @YourBot summarize this thread
   ```

2. The bot will:
   - React with a ⏳ emoji to show it's processing
   - Read all messages in the thread
   - Generate a comprehensive summary focusing on:
     - Main topics discussed
     - Key decisions or conclusions
     - Action items or next steps
     - Overall sentiment and tone
   - Post the summary as a reply in the thread
   - React with a ✅ emoji when complete

## Setup

1. Install dependencies using the virtual environment:
   ```
   source slackMcp/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   - Copy `.env-example` to `.env` and fill in your Slack tokens and LLM API keys
   - Copy `mcp_agent.secrets.yaml.example` to `mcp_agent.secrets.yaml` and fill in your LLM API keys

3. Running the bot:
   - For the full featured MCP bot:
     ```
     cd mcp_simple_slackbot
     python main.py
     ```
   - For just the thread summarizer bot:
     ```
     python thread_summarizer.py
     ```

## MCP-Agent Configuration

The thread summarization feature uses the mcp-agent library, which requires configuration via YAML files:

- `mcp_agent.config.yaml`: General configuration (already set up)
- `mcp_agent.secrets.yaml`: API keys (you need to create this from the example)

## Testing

To test the thread summarization functionality without connecting to Slack:

```
cd mcp_simple_slackbot
python test_summarization.py
```

This script simulates a thread conversation and generates a summary using the same code that powers the thread summarization feature.

## Technical Implementation

The thread summarization feature uses:
- `mcp-agent` library for agent workflows
- `Agent` class with `OpenAIAugmentedLLM` or `AnthropicAugmentedLLM` for LLM integration
- Contextual prompting with `RequestParams` for high-quality summaries
- Slack conversations_replies API to fetch thread messages

## Requirements

- Python 3.9+
- Slack Bot & App tokens with appropriate permissions
- API key for at least one LLM provider (OpenAI or Anthropic)
- MCP servers as configured in `servers_config.json`
