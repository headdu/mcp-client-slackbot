#!/usr/bin/env python3
"""
Test script for the thread summarization feature.
This script simulates thread messages and tests the summary generation.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

# MCP-Agent imports
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Sample thread messages for testing
SAMPLE_THREAD = [
    {"user": "U1234567", "text": "Hey team, I think we should discuss the roadmap for Q3. I have some ideas I'd like to share."},
    {"user": "U2345678", "text": "Sounds good! What are you thinking?"},
    {"user": "U1234567", "text": "I believe we should focus on three main areas: 1) Improving the user onboarding experience, 2) Adding the analytics dashboard, and 3) Fixing the performance issues on the mobile app."},
    {"user": "U3456789", "text": "I agree with points 1 and 3, but I'm not sure about the analytics dashboard. Do we have enough resources for that?"},
    {"user": "U4567890", "text": "The analytics dashboard is actually not that complex. I think we can handle it within 3-4 weeks with 2 engineers."},
    {"user": "U2345678", "text": "What about the backend infrastructure upgrade we discussed last month? Shouldn't that be a priority too?"},
    {"user": "U1234567", "text": "You're right, I forgot about that. Perhaps we can start with the infrastructure upgrade and user onboarding in the first month, then move to the analytics and mobile app in months 2-3?"},
    {"user": "U3456789", "text": "That sounds like a reasonable plan. Let's put together a detailed timeline and resource allocation."},
    {"user": "U4567890", "text": "I can work on the timeline this week and share it with everyone by Friday. @U2345678 can you help with resource allocation?"},
    {"user": "U2345678", "text": "Sure, I'll coordinate with the team leads and have something by Thursday."},
    {"user": "U1234567", "text": "Great! Let's reconvene next Monday to finalize the Q3 roadmap then."}
]


def format_thread_for_summarization(messages):
    """Format thread messages for summarization."""
    formatted_text = "Here is the conversation thread:\n\n"

    for i, msg in enumerate(messages):
        user_id = msg.get("user", "unknown")
        text = msg.get("text", "").strip()
        formatted_text += f"Message {i+1} - User {user_id}:\n{text}\n\n"

    return formatted_text


async def summarize_thread(thread_text):
    """Generate a summary using mcp-agent."""
    load_dotenv()  # Load API keys from .env file

    # Check for API keys
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("No API key found in environment variables")

    # Initialize MCP app
    mcp_app = MCPApp()

    # Create an agent for the summarization task
    summarizer_agent = Agent(
        name="thread_summarizer",
        instruction="You analyze Slack conversation threads and provide concise summaries.",
        server_names=[]  # No server needs for basic summarization
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

    logger.info("Generating summary...")

    # Create request parameters for better results
    request_params = RequestParams(
        modelPreferences=ModelPreferences(
            costPriority=0.3,         # Balance cost with quality
            speedPriority=0.2,        # Not super urgent
            intelligencePriority=0.5  # Higher priority on intelligence for better summaries
        )
    )

    async with mcp_app.run(), summarizer_agent:
        # Choose LLM based on available API keys
        if os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            # Use OpenAI if only OpenAI key is available
            llm = await summarizer_agent.attach_llm(OpenAIAugmentedLLM)
        else:
            # Default to Anthropic (Claude) if available or if both are available
            llm = await summarizer_agent.attach_llm(AnthropicAugmentedLLM)

        summary = await llm.generate_str(
            message=prompt,
            request_params=request_params
        )

    return summary


async def main():
    """Run the test."""
    try:
        # Format the sample thread
        thread_text = format_thread_for_summarization(SAMPLE_THREAD)

        # Print the formatted thread
        logger.info("Sample thread for testing:")
        print("\n" + "-" * 50)
        print(thread_text)
        print("-" * 50 + "\n")

        # Generate and print the summary
        summary = await summarize_thread(thread_text)

        logger.info("Generated summary:")
        print("\n" + "=" * 50)
        print(summary)
        print("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
