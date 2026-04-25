import asyncio
import os
import sys
from typing import Any

import dotenv
from github import Github

from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI


dotenv.load_dotenv()


# GitHub Actions passes these as environment variables.
# Fallback to argv is included because the workflow command may also pass them as arguments.
if len(sys.argv) >= 6:
    os.environ.setdefault("GITHUB_TOKEN", sys.argv[1])
    os.environ.setdefault("REPOSITORY", sys.argv[2])
    os.environ.setdefault("PR_NUMBER", sys.argv[3])
    os.environ.setdefault("OPENAI_API_KEY", sys.argv[4])
    os.environ.setdefault("OPENAI_BASE_URL", sys.argv[5])


github_token = os.getenv("GITHUB_TOKEN")
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")


git = Github(github_token) if github_token else Github()

if not repository:
    raise RuntimeError("REPOSITORY environment variable is missing.")

if not pr_number:
    raise RuntimeError("PR_NUMBER environment variable is missing.")

repo = git.get_repo(repository)


llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """
    Fetch pull request details by PR number.

    Returns the PR author, title, body, diff URL, state,
    head SHA, base SHA, and commit SHAs.
    """
    pull_request = repo.get_pull(int(pr_number))

    commit_shas: list[str] = []

    for commit in pull_request.get_commits():
        commit_shas.append(commit.sha)

    return {
        "number": pull_request.number,
        "user": pull_request.user.login if pull_request.user else None,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "base_sha": pull_request.base.sha,
        "commit_SHAs": commit_shas,
    }


def get_pr_changed_files(pr_number: int) -> list[dict[str, Any]]:
    """
    Fetch all changed files in a pull request by PR number.

    Use this tool when reviewing a PR or when the user asks which files changed.
    This returns files changed across the whole PR, not just the latest commit.
    """
    pull_request = repo.get_pull(int(pr_number))

    changed_files: list[dict[str, Any]] = []

    for file in pull_request.get_files():
        changed_files.append(
            {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch,
            }
        )

    return changed_files


def get_file_contents(file_path: str) -> dict[str, Any]:
    """
    Fetch the contents of a file from the repository by file path.

    If the exact path is not found, search the repository recursively
    for a file with the same name.
    """

    def read_file(path: str) -> dict[str, Any]:
        file_content = repo.get_contents(path)

        if isinstance(file_content, list):
            return {
                "path": path,
                "error": "The provided path is a directory, not a file.",
            }

        return {
            "path": path,
            "content": file_content.decoded_content.decode("utf-8"),
        }

    try:
        return read_file(file_path)

    except Exception:
        target_name = file_path.split("/")[-1]

        try:
            tree = repo.get_git_tree(repo.default_branch, recursive=True)

            matches = [
                item.path
                for item in tree.tree
                if item.type == "blob" and item.path.split("/")[-1] == target_name
            ]

            if not matches:
                return {
                    "path": file_path,
                    "error": f"File not found: {file_path}",
                }

            return read_file(matches[0])

        except Exception as exc:
            return {
                "path": file_path,
                "error": str(exc),
            }


def get_pr_commit_details(head_sha: str) -> dict[str, Any]:
    """
    Fetch commit details by commit SHA.

    Returns the commit SHA, commit message, author,
    and files changed in that specific commit.
    """
    commit = repo.get_commit(head_sha)

    changed_files: list[dict[str, Any]] = []

    for file in commit.files:
        changed_files.append(
            {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch,
            }
        )

    return {
        "sha": commit.sha,
        "message": commit.commit.message,
        "author": commit.commit.author.name if commit.commit.author else None,
        "changed_files": changed_files,
    }


async def add_context_to_state(ctx: Context, context: str) -> str:
    """
    Save gathered pull request and repository context to the workflow state.
    """
    current_state = await ctx.store.get("state", default={})

    previous_context = current_state.get("gathered_contexts", "")

    if previous_context:
        current_state["gathered_contexts"] = f"{previous_context}\n\n{context}"
    else:
        current_state["gathered_contexts"] = context

    await ctx.store.set("state", current_state)

    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """
    Save the drafted pull request review comment to the workflow state.
    """
    current_state = await ctx.store.get("state", default={})

    current_state["draft_comment"] = draft_comment
    current_state["review_comment"] = draft_comment

    await ctx.store.set("state", current_state)

    return (
        "State updated with draft review comment. "
        "Now hand off to ReviewAndPostingAgent for final review and posting."
    )


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """
    Save the final reviewed pull request comment to the workflow state.
    """
    current_state = await ctx.store.get("state", default={})

    current_state["final_review"] = final_review

    await ctx.store.set("state", current_state)

    return "State updated with final review."


def post_review_to_github(pr_number: int, comment: str) -> dict[str, Any]:
    """
    Post the final review comment to GitHub as a pull request review.
    """
    pull_request = repo.get_pull(int(pr_number))

    review = pull_request.create_review(
        body=comment,
    )

    return {
        "status": "posted",
        "pr_number": int(pr_number),
        "review_id": review.id,
        "body": comment,
    }


pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
pr_changed_files_tool = FunctionTool.from_defaults(fn=get_pr_changed_files)
file_contents_tool = FunctionTool.from_defaults(fn=get_file_contents)
pr_commit_details_tool = FunctionTool.from_defaults(fn=get_pr_commit_details)
post_review_to_github_tool = FunctionTool.from_defaults(fn=post_review_to_github)

add_context_to_state_tool = FunctionTool.from_defaults(
    async_fn=add_context_to_state,
    name="add_context_to_state",
    description="Save gathered pull request and repository context to workflow state.",
)

add_comment_to_state_tool = FunctionTool.from_defaults(
    async_fn=add_comment_to_state,
    name="add_comment_to_state",
    description="Save the drafted pull request review comment to workflow state.",
)

add_final_review_to_state_tool = FunctionTool.from_defaults(
    async_fn=add_final_review_to_state,
    name="add_final_review_to_state",
    description="Save the final reviewed pull request comment to workflow state.",
)


context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description=(
        "Gathers all needed GitHub repository and pull request context, "
        "including PR details, changed files, commit details, and requested file contents."
    ),
    tools=[
        pr_details_tool,
        pr_changed_files_tool,
        file_contents_tool,
        pr_commit_details_tool,
        add_context_to_state_tool,
    ],
    system_prompt=(
        "You are the context gathering agent. When gathering context, you MUST gather:\n"
        "- PR details: author, title, body, diff_url, state, and head_sha using get_pr_details;\n"
        "- Changed files for the whole pull request using get_pr_changed_files. "
        "You MUST call get_pr_changed_files when reviewing a PR;\n"
        "- Any requested files using get_file_contents.\n\n"
        "After gathering the information, you MUST save a clear summary using add_context_to_state. "
        "The saved context MUST include the PR number and changed file names exactly.\n\n"
        "Once you save the context, hand control back to the CommentorAgent."
    ),
    can_handoff_to=["CommentorAgent"],
)


commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description=(
        "Uses the context gathered by the context agent to draft a pull request review comment. "
        "This agent must save the draft comment and then hand off to ReviewAndPostingAgent."
    ),
    tools=[
        add_comment_to_state_tool,
    ],
    system_prompt=(
        "You are the commentor agent that writes review comments for pull requests as a human reviewer would.\n\n"
        "IMPORTANT WORKFLOW RULES:\n"
        "- You MUST NOT finish the workflow yourself.\n"
        "- You MUST NOT return the review as the final answer to the user.\n"
        "- You MUST save the drafted review using add_comment_to_state.\n"
        "- After saving the draft review, you MUST hand off to ReviewAndPostingAgent.\n"
        "- The handoff to ReviewAndPostingAgent is mandatory, even if the draft looks complete.\n\n"
        "Review writing instructions:\n"
        "- Request the PR details, changed files, and any other repo files you may need from the ContextAgent.\n"
        "- Once you have the needed information, write a good ~200-300 word review in markdown format detailing:\n"
        "  - What is good about the PR?\n"
        "  - Did the author follow ALL contribution rules? What is missing?\n"
        "  - Are there tests for new functionality?\n"
        "  - If there are new models, are there migrations for them? Use the diff to determine this.\n"
        "  - Are new endpoints documented? Use the diff to determine this.\n"
        "  - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.\n"
        "- You should directly address the author.\n\n"
        "Required sequence:\n"
        "1. If context is missing, hand off to ContextAgent.\n"
        "2. After ContextAgent returns, draft the review.\n"
        "3. Call add_comment_to_state with the draft review.\n"
        "4. Immediately call handoff to ReviewAndPostingAgent with reason: "
        "'Draft review saved and ready for final review and posting.'\n"
    ),
    can_handoff_to=[
        "ContextAgent",
        "ReviewAndPostingAgent",
    ],
)


review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description=(
        "Reviews the drafted pull request comment, requests rewrites if needed, "
        "saves the final review, and posts it to GitHub."
    ),
    tools=[
        add_final_review_to_state_tool,
        post_review_to_github_tool,
    ],
    system_prompt=(
        "You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.\n"
        "Once a review is generated, you need to run a final check and post it to GitHub.\n"
        "The review must:\n"
        "- Be a ~200-300 word review in markdown format.\n"
        "- Specify what is good about the PR.\n"
        "- Check whether the author followed ALL contribution rules and what is missing.\n"
        "- Include notes on test availability for new functionality.\n"
        "- If there are new models, include notes on whether migrations are present.\n"
        "- Include notes on whether new endpoints were documented.\n"
        "- Include suggestions on which lines could be improved upon, with quoted lines.\n\n"
        "If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.\n"
        "When you are satisfied, you MUST save the final review using add_final_review_to_state.\n"
        "Then you MUST post the review to GitHub using post_review_to_github.\n"
        "When calling post_review_to_github, use the PR number from the user's request."
    ),
    can_handoff_to=[
        "CommentorAgent",
    ],
)


workflow_agent = AgentWorkflow(
    agents=[
        context_agent,
        commentor_agent,
        review_and_posting_agent,
    ],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "review_comment": "",
        "final_review": "",
    },
)


async def main():
    query = (
        f"Write a review for PR number {pr_number}. "
        f"Gather the needed context, draft the review, run final review checks, "
        f"and post the final review to GitHub."
    )

    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None

    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")

        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)

            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])

        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")

        elif isinstance(event, ToolCall):
            print(
                f"Calling selected tool: {event.tool_name}, "
                f"with arguments: {event.tool_kwargs}"
            )


if __name__ == "__main__":
    asyncio.run(main())
    git.close()