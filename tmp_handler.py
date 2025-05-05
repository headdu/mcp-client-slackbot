async def handle_github_issue_creation(self, event, say):
    """Handle creating a GitHub issue from a thread using the dedicated agent."""
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
        else:
            # Issue creation failed
            await say(
                text=f"❌ Failed to create GitHub issue: {result.get('error', 'Unknown error')}",
                thread_ts=thread_ts
            )

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
