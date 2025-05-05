# MCP Simple Slackbot

A Slack bot that integrates with Model Context Protocol (MCP) servers to enable AI-powered interactions and thread summarization.

## Features

- Integrates with various MCP servers (fetch, sqlite, git, github, etc.)
- Supports multiple LLM providers (OpenAI, Anthropic, Groq)
- Handles direct messages and mentions
- Executes tools through MCP servers
- **Thread Summarization**: Summarizes Slack conversation threads on demand

## Thread Summarization

The bot can now summarize Slack conversation threads! This feature is powered by the mcp-agent library, which provides advanced agent workflows for complex tasks.

### How to use thread summarization:

1. In any Slack thread, mention the bot with the word "summarize"
   ```
   @MCP Assistant Can you summarize this thread?
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

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
   SLACK_BOT_TOKEN=xoxb-your-bot-token
   SLACK_APP_TOKEN=xapp-your-app-token
   ANTHROPIC_API_KEY=your-anthropic-key (or OpenAI/Groq key)
   LLM_MODEL=claude-3-7-sonnet-20250219 (or other model)
   ```
4. Configure MCP servers in `servers_config.json`
5. Run the bot:
   ```
   python main.py
   ```

## Technical Implementation

The thread summarization feature uses:
- `mcp-agent` library for agent workflows
- AugmentedLLM workflow for summarization
- Slack conversations_replies API to fetch thread messages
- Contextual prompting to generate high-quality summaries

## Requirements

- Python 3.9+
- Slack Bot & App tokens with appropriate permissions
- API key for at least one LLM provider
- MCP servers as configured in `servers_config.json`
