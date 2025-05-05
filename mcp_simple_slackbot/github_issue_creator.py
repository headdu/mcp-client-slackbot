import asyncio
import logging
from typing import Dict, Any

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GitHubIssueCreator:
    """Agent for creating GitHub issues from Slack thread content."""

    def __init__(self) -> None:
        """Initialize the GitHub issue creator agent."""
        self.app = MCPApp(name="github_issue_creator")
        self.agent = None

    async def initialize(self) -> None:
        """Initialize the agent and connect to servers."""
        self.agent = Agent(
            name="github_issue_creator_agent",
            instruction="""You are an agent that creates GitHub issues from Slack thread content.
            Your task is to:
            1. Analyze the thread content to extract a meaningful issue title and description
            2. Format the content appropriately for GitHub
            3. Create the issue in the specified GitHub repository
            4. Return the URL of the created issue

            The issue should be well-formatted, with a clear title that summarizes the problem
            and a detailed description that provides all necessary context from the thread.
            """,
            server_names=["github"],
        )
        logging.info("GitHub issue creator agent initialized")

    async def create_issue(
        self, thread_content: str, owner: str, repo: str, use_anthropic: bool = True
    ) -> Dict[str, Any]:
        """Create a GitHub issue from thread content.

        Args:
            thread_content: The formatted content of the Slack thread
            owner: The GitHub repository owner
            repo: The GitHub repository name
            use_anthropic: Whether to use Anthropic (True) or OpenAI (False)

        Returns:
            Dictionary containing issue information including URL
        """
        try:
            if not self.agent:
                await self.initialize()

            async with self.agent:
                # Attach appropriate LLM
                if use_anthropic:
                    llm = await self.agent.attach_llm(AnthropicAugmentedLLM)
                else:
                    llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                # Configure request parameters
                request_params = RequestParams(
                    modelPreferences=ModelPreferences(
                        costPriority=0.3,
                        speedPriority=0.2,
                        intelligencePriority=0.5
                    )
                )

                # Prepare the prompt
                prompt = f"""
                I need you to create a GitHub issue based on a Slack thread conversation.

                First, analyze this Slack thread to extract the key information:

                {thread_content}

                Based on this thread:

                1. Create a clear, concise title that summarizes the main point
                2. Create a detailed description that captures all relevant information from the thread
                3. Use the GitHub server to create an issue in the repository {owner}/{repo}
                    - Do it with the following template
                    Everytime you're asked to create an issue and you're describing a bug, use the following template

                    ```
                    ## Issue description
                    <!-- Provide a clear description of the issue -->


                    <!-- Ideally provide video description and/or screenshots -->


                    <!-- Provide links to any relevant Slack conversation -->


                    ## Steps to reproduce
                    1. 
                    2. 

                    ## Expected behavior
                    <!-- A clear and concise description of what you expected to happen instead -->

                    ## Environment
                    - Browser: [e.g. chrome, safari]
                    - Version: [e.g. 22]
                    - Device: [e.g. desktop or mobile]

                    ## Technical refinement

                    ## Testing instructions
                    <!-- To be filled by developers in case there are any special considerations for testing this bug -->

                    <!--  ====>>> Assign testers and change testing status to Ready to Test <<<====  -->

                    ## Definition of done
                    - [ ] matches expected behaviour
                    - [ ] passes CI workflow
                    - [ ] code reviewed
                    - [ ] accepted by testers
                    - [ ] deployed to production
                    ```


                    Usually when you're asked to create a bug, it will come from a slack thread.
                    When it does, and the slack thread contains a human readable ID on top (like 1500) include that number in the issue title.
                    like

                    title - (1500)
                4. Return the URL of the created issue and its details

                Format your interactions and the final issue properly for GitHub Markdown.
                """

                # Generate the issue
                response = await llm.generate_str(
                    message=prompt,
                    request_params=request_params
                )

                # Extract the issue URL from the response if it exists
                issue_url = None
                if "http" in response:
                    for word in response.split():
                        if word.startswith("http") and "github.com" in word and "/issues/" in word:
                            issue_url = word.strip(",.;:()")
                            break

                return {
                    "success": True,
                    "response": response,
                    "issue_url": issue_url
                }

        except Exception as e:
            logging.error(f"Error creating GitHub issue: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.agent:
            await self.agent.close()
            logging.info("GitHub issue creator agent closed")


async def create_github_issue(
    thread_content: str, owner: str, repo: str, use_anthropic: bool = True
) -> Dict[str, Any]:
    """Standalone function to create a GitHub issue from thread content.

    Args:
        thread_content: The formatted content of the Slack thread
        owner: The GitHub repository owner
        repo: The GitHub repository name
        use_anthropic: Whether to use Anthropic (True) or OpenAI (False)

    Returns:
        Dictionary with issue information
    """
    creator = GitHubIssueCreator()
    try:
        return await creator.create_issue(
            thread_content=thread_content,
            owner=owner,
            repo=repo,
            use_anthropic=use_anthropic
        )
    finally:
        await creator.cleanup()


if __name__ == "__main__":
    # Example for testing
    async def run_test():
        test_thread = """
        Message 1 - @user1:
        We need to fix the login page on the website. Users are reporting that they sometimes get a 500 error when trying to log in with their Google account.

        Message 2 - @user2:
        I've seen this too. It seems to happen more often when the server is under high load.

        Message 3 - @user3:
        The logs show a timeout when connecting to the OAuth provider. We should add better error handling and retry logic.

        Message 4 - @user1:
        Let's create an issue for this and prioritize it for the next sprint.
        """

        result = await create_github_issue(
            thread_content=test_thread,
            owner="test-owner",
            repo="test-repo"
        )

        print("Result:", result)

    asyncio.run(run_test())
