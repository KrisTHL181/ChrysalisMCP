from typing import Optional, List, Dict, Callable, Awaitable
import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request
from starlette.responses import Response
import uvicorn
import inspect
import json
import os
from pathlib import Path
import mimetypes
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import re

# RAG Configuration
FAISS_INDEX_PATH = "faiss_index"
CHUNKS_PATH = "chunks.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global RAG components
rag_vectorstore = None
rag_embeddings = None
bm25_retriever = None
all_chunks = []
global_memory_content = ""


async def load_rag_components():
    global rag_vectorstore, rag_embeddings, bm25_retriever, all_chunks, global_memory_content
    logger.info("Loading RAG components...")
    try:
        rag_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        rag_vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, rag_embeddings, allow_dangerous_deserialization=True
        )
        logger.info("FAISS vector store loaded.")

        # Load all chunks for BM25
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            all_chunks_data = json.load(f)
        all_chunks = [d["page_content"] for d in all_chunks_data]
        tokenized_corpus = [doc.split(" ") for doc in all_chunks]
        bm25_retriever = BM25Okapi(tokenized_corpus)
        logger.info("BM25 retriever initialized.")

        # Load global memory content
        global_memory_file = os.path.expanduser("~/.ChrysalisMCP/global_memory.md")
        if os.path.exists(global_memory_file):
            with open(global_memory_file, "r", encoding="utf-8") as f:
                global_memory_content = f.read()
            logger.info("Global memory loaded.")
        else:
            logger.warning(
                "Global memory file not found. It will be created on first update."
            )

        logger.info("RAG components loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load RAG components: {e}")
        rag_vectorstore = None
        rag_embeddings = None
        bm25_retriever = None
        all_chunks = []
        global_memory_content = ""


# --- Resource Definitions ---
RESOURCE_BASE_DIR = Path("./mcp_resources").resolve()

# Global flag for security
ALLOW_ABSOLUTE_PATHS = False  # Will be set by CLI option

# Import the default_api which provides access to the CLI tools
import default_api as default_api_module
import mcp_calculator

# --- Tool Decorator and Registration ---
_tools: Dict[str, types.Tool] = {}
_tool_implementations: Dict[str, Callable[..., Awaitable[str]]] = {}


def tool(name: str, title: str):
    """Decorator to register a function as a tool."""

    def decorator(func: Callable[..., Awaitable[str]]):
        # Extract schema from signature and docstring
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)

        # Parse the docstring to get parameter descriptions
        param_descriptions = {}
        if docstring:
            arg_section = re.search(r"Args:\n((.|\n)*)", docstring)
            if arg_section:
                arg_lines = arg_section.group(1).strip().split("\n")
                for line in arg_lines:
                    match = re.match(r"\s*(\w+)\s*:\s*(.*)", line)
                    if match:
                        param_name, param_desc = match.groups()
                        param_descriptions[param_name.strip()] = param_desc.strip()

        properties = {}
        required = []
        type_mapping = {
            str: "string",
            int: "number",
            float: "number",
            bool: "boolean",
            dict: "object",
            list: "array",
            None: "null",
        }

        for param_name, param in signature.parameters.items():
            if param_name in ("self", "name", "arguments"):
                continue

            param_type = "string"
            is_optional = False
            items_type = "string"

            if hasattr(param.annotation, "__origin__"):
                if param.annotation.__origin__ is Optional:
                    is_optional = True
                    inner_type = param.annotation.__args__[0]
                    if (
                        hasattr(inner_type, "__origin__")
                        and inner_type.__origin__ is list
                    ):
                        param_type = "array"
                        if inner_type.__args__:
                            items_type = type_mapping.get(
                                inner_type.__args__[0], "string"
                            )
                    else:
                        param_type = type_mapping.get(inner_type, "string")
                elif param.annotation.__origin__ is list:
                    param_type = "array"
                    if param.annotation.__args__:
                        items_type = type_mapping.get(
                            param.annotation.__args__[0], "string"
                        )
                else:
                    param_type = type_mapping.get(param.annotation, "string")
            elif param.annotation is not inspect.Parameter.empty:
                param_type = type_mapping.get(param.annotation, "string")
            elif param.default is not inspect.Parameter.empty:
                param_type = type_mapping.get(type(param.default), "string")

            description = param_descriptions.get(param_name, f"Parameter {param_name}")
            if param_type == "array":
                properties[param_name] = {
                    "type": "array",
                    "items": {"type": items_type},
                    "description": description,
                }
            else:
                properties[param_name] = {
                    "type": param_type,
                    "description": description,
                }

            if param.default is inspect.Parameter.empty and not is_optional:
                required.append(param_name)

        # Extract tool description from the first line of the docstring
        description = (
            docstring.split("\n")[0] if docstring else "No description available."
        )

        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        _tools[name] = types.Tool(
            name=title, description=description, inputSchema=input_schema
        )
        _tool_implementations[name] = func
        return func

    return decorator


app = Server("mcp-cli-tools-server")

# --- Prompt Definitions ---
PROMPTS = {
    "generate-commit-message": types.Prompt(
        name="generate-commit-message",
        description="Generate a Git commit message based on changes.",
        arguments=[
            types.PromptArgument(
                name="changes",
                description="Description of code changes or git diff output.",
                required=True,
            )
        ],
    ),
    "explain-code": types.Prompt(
        name="explain-code",
        description="Explain how a given code snippet works.",
        arguments=[
            types.PromptArgument(
                name="code", description="The code snippet to explain.", required=True
            ),
            types.PromptArgument(
                name="language",
                description="The programming language of the code (e.g., python, javascript).",
                required=False,
            ),
        ],
    ),
    "summarize-text": types.Prompt(
        name="summarize-text",
        description="Summarize a given text.",
        arguments=[
            types.PromptArgument(
                name="text", description="The text to summarize.", required=True
            ),
            types.PromptArgument(
                name="length",
                description="Desired length of the summary (e.g., short, medium, long).",
                required=False,
            ),
        ],
    ),
    "generate-code": types.Prompt(
        name="generate-code",
        description="Generate code based on a description.",
        arguments=[
            types.PromptArgument(
                name="description",
                description="Description of the code to generate.",
                required=True,
            ),
            types.PromptArgument(
                name="language",
                description="The programming language for the generated code.",
                required=False,
            ),
        ],
    ),
    "ask-rag": types.Prompt(
        name="ask-rag",
        description="Ask a question that can be answered by retrieving information from the local knowledge base.",
        arguments=[
            types.PromptArgument(
                name="query", description="The question to ask.", required=True
            )
        ],
    ),
    "generate-clue": types.Prompt(
        name="generate-clue",
        description="Generate a retrieval clue based on a query and global memory.",
        arguments=[
            types.PromptArgument(
                name="query", description="The original user query.", required=True
            ),
            types.PromptArgument(
                name="global_memory",
                description="Summary of the global knowledge base.",
                required=False,
            ),
        ],
    ),
}


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return list(PROMPTS.values())


@app.get_prompt()
async def get_prompt(
    name: str, arguments: dict[str, str] | None = None
) -> types.GetPromptResult:
    if name not in PROMPTS:
        raise ValueError(f"Prompt not found: {name}")

    if name == "generate-commit-message":
        changes = arguments.get("changes", "") if arguments else ""
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Generate a concise and descriptive Git commit message for the following changes:\n\n{changes}",
                    ),
                )
            ]
        )

    if name == "explain-code":
        code = arguments.get("code", "") if arguments else ""
        language = arguments.get("language", "Unknown") if arguments else "Unknown"
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Explain how this {language} code works:\n\n```\n{code}\n```",
                    ),
                )
            ]
        )

    if name == "summarize-text":
        text = arguments.get("text", "") if arguments else ""
        length = arguments.get("length", "") if arguments else ""
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Summarize the following text. Desired length: {length}.\n\n{text}",
                    ),
                )
            ]
        )

    if name == "generate-code":
        description = arguments.get("description", "") if arguments else ""
        language = arguments.get("language", "") if arguments else ""
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Generate {language} code for the following description:\n\n{description}",
                    ),
                )
            ]
        )

    if name == "ask-rag":
        query = arguments.get("query", "") if arguments else ""
        clue_query = (
            arguments.get("clue_query", "") if arguments else ""
        )  # Add clue_query argument

        if not query:
            raise ValueError("Missing required argument 'query' for ask-rag")

        if not clue_query:  # If clue_query is not provided, generate a clue
            clue_prompt_text = f"""Based on the following original query and global memory summary, generate a more detailed and comprehensive retrieval query (clue) that would help find relevant information. The clue should be a single, concise query.\n\nOriginal Query: {query}\nGlobal Memory Summary: {global_memory_content}\n\nGenerated Clue:"""
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=clue_prompt_text),
                    )
                ]
            )

        if rag_vectorstore is None or bm25_retriever is None:
            logger.error("RAG components not loaded. Cannot answer RAG query.")
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text="RAG knowledge base is not available. Please inform the developer.",
                        ),
                    )
                ]
            )

        logger.info(
            f"Performing RAG search for query: {query} with clue: {clue_query}"
        )  # Log clue_query

        # Semantic search (FAISS)
        faiss_docs = rag_vectorstore.similarity_search(
            clue_query, k=5
        )  # Use clue_query for search
        faiss_context = "\n\n".join([doc.page_content for doc in faiss_docs])
        logger.debug(f"Retrieved FAISS context:\n{faiss_context}")

        # Keyword search (BM25)
        tokenized_query = clue_query.split(" ")  # Use clue_query for BM25
        bm25_scores = bm25_retriever.get_scores(tokenized_query)
        # Get top 5 BM25 documents based on scores
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:5]
        bm25_docs = [all_chunks[i] for i in top_bm25_indices]
        bm25_context = "\n\n".join(bm25_docs)
        logger.debug(f"Retrieved BM25 context:\n{bm25_context}")

        # Combine contexts (simple concatenation for now, can be refined)
        combined_context = f"Semantic Search Results:\n{faiss_context}\n\nKeyword Search Results:\n{bm25_context}"

        # Construct the prompt with retrieved context
        rag_prompt = f"""Based on the following information, answer the question. If the information does not contain the answer, state that you don't know.\n\nGlobal Memory Summary:\n{global_memory_content}\n\nContext:\n{combined_context}\n\nQuestion: {query}\nAnswer:"""

        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user", content=types.TextContent(type="text", text=rag_prompt)
                )
            ]
        )

    if name == "generate-clue":
        query = arguments.get("query", "") if arguments else ""
        global_memory = arguments.get("global_memory", "") if arguments else ""

        clue_prompt_text = f"""Based on the following original query and global memory summary, generate a more detailed and comprehensive retrieval query (clue) that would help find relevant information. The clue should be a single, concise query.\n\nOriginal Query: {query}\nGlobal Memory Summary: {global_memory}\n\nGenerated Clue:"""

        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=clue_prompt_text),
                )
            ]
        )

    raise ValueError("Prompt implementation not found")


# Helper for path validation
def validate_path_for_mcp_tool(
    input_path: str, is_write_operation: bool = False
) -> Path:
    resolved_path = Path(input_path).resolve()
    current_working_dir = Path(os.getcwd()).resolve()

    if ALLOW_ABSOLUTE_PATHS:
        # If allowed, just return the resolved path
        return resolved_path
    else:
        # If not allowed, path must be within CWD or RESOURCE_BASE_DIR
        try:
            # Check if path is relative to CWD
            resolved_path.relative_to(current_working_dir)
            return resolved_path
        except ValueError:
            try:
                # Check if path is relative to RESOURCE_BASE_DIR
                resolved_path.relative_to(RESOURCE_BASE_DIR)
                return resolved_path
            except ValueError:
                raise ValueError(
                    f"Access denied: Path '{input_path}' is outside allowed directories (current working directory or mcp_resources)."
                )


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    resources = []
    # Ensure the base directory exists
    RESOURCE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Dynamically list files in the resource base directory
    for file_path in RESOURCE_BASE_DIR.rglob("*"):  # rglob for recursive search
        if file_path.is_file():
            relative_path = file_path.relative_to(RESOURCE_BASE_DIR)
            uri = f"file:///{relative_path.as_posix()}"  # Use posix path for URI consistency
            name = relative_path.name
            mime_type, _ = mimetypes.guess_type(file_path.name)
            if mime_type is None:
                mime_type = "application/octet-stream"  # Default to generic binary if type cannot be guessed

            resources.append(
                types.Resource(
                    uri=uri,
                    name=name,
                    mimeType=mime_type,
                    description=f"Dynamically discovered resource: {relative_path}",
                )
            )
    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    # Extract the relative path from the URI
    if not uri.startswith("file:///"):
        raise ValueError(f"Unsupported resource URI scheme: {uri}")

    relative_path_str = uri[len("file:///") :]
    file_path = RESOURCE_BASE_DIR / relative_path_str

    if not file_path.is_file() or not file_path.exists():
        raise ValueError(f"Resource not found: {uri}")

    # Ensure the file is within the allowed base directory
    try:
        file_path.relative_to(RESOURCE_BASE_DIR)
    except ValueError:
        raise ValueError(
            f"Access denied: {uri} is outside the allowed resource directory."
        )

    # Use the default_api_module.read_file to read the content
    # Note: default_api_module.read_file currently returns text.
    # For binary files, you would need to handle base64 encoding.
    result = default_api_module.read_file(absolute_path=str(file_path))
    if "error" in result:
        raise ValueError(f"Failed to read resource {uri}: {result["error"]}")
    return result["output"]


# --- Tool Implementations ---


@app.call_tool()
async def call_tool_handler(name: str, arguments: dict) -> str:
    if name not in _tool_implementations:
        raise ValueError(f"Unknown tool: {name}")

    # Special path validation logic
    if name in [
        "read_file",
        "write_file",
        "list_directory",
        "search_file_content",
        "glob",
        "replace",
        "read_many_files",
    ]:
        path_arg_names = ["absolute_path", "file_path", "path", "paths"]
        for arg_name in path_arg_names:
            if arg_name in arguments:
                is_write = name in ["write_file", "replace"]
                if isinstance(arguments[arg_name], list):
                    validated_paths = [
                        str(validate_path_for_mcp_tool(p, is_write))
                        for p in arguments[arg_name]
                    ]
                    arguments[arg_name] = validated_paths
                else:
                    validated_path = validate_path_for_mcp_tool(
                        arguments[arg_name], is_write
                    )
                    arguments[arg_name] = str(validated_path)

    # Call the actual tool implementation
    implementation = _tool_implementations[name]
    result = await implementation(**arguments)
    return result


@tool(name="calculate", title="Calculator")
async def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    Args:
        expression: The mathematical expression to evaluate.
    """
    try:
        result = mcp_calculator.calculate(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(name="google_search", title="Google Search")
async def google_search(query: str) -> str:
    """
    Performs a web search using Google Search.
    Args:
        query: The search query to find information on the web.
    """
    result = default_api_module.google_web_search(query=query)
    return json.dumps(result)


@tool(name="web_fetch", title="Web Fetch")
async def web_fetch(prompt: str) -> str:
    """
    Fetches content from URLs specified in the prompt.
    Args:
        prompt: A comprehensive prompt that includes the URL(s) to fetch.
    """
    result = default_api_module.web_fetch(prompt=prompt)
    return json.dumps(result)


@tool(name="shell", title="Shell Command")
async def shell(
    command: str, description: Optional[str] = None, directory: Optional[str] = None
) -> str:
    """
    Executes a shell command.
    Args:
        command: The exact shell command to execute.
        description: A brief description of the command's purpose.
        directory: The directory to run the command in.
    """
    args = {"command": command, "description": description, "directory": directory}
    # Filter out None values so we don't pass them to the underlying function
    args = {k: v for k, v in args.items() if v is not None}
    result = default_api_module.run_shell_command(**args)
    return json.dumps(result)


@tool(name="read_file", title="Read File")
async def read_file(
    absolute_path: str, limit: Optional[float] = None, offset: Optional[float] = None
) -> str:
    """
    Reads content from a specified file.
    Args:
        absolute_path: The absolute path of the file to read.
        limit: Maximum number of lines to read.
        offset: Line number to start reading from.
    """
    args = {"absolute_path": absolute_path, "limit": limit, "offset": offset}
    args = {k: v for k, v in args.items() if v is not None}
    result = default_api_module.read_file(**args)
    return json.dumps(result)


@tool(name="write_file", title="Write File")
async def write_file(file_path: str, content: str) -> str:
    """
    Writes content to a specified file.
    Args:
        file_path: The absolute path of the file to write to.
        content: The content to write into the file.
    """
    result = default_api_module.write_file(file_path=file_path, content=content)
    return json.dumps(result)


@tool(name="list_directory", title="List Directory")
async def list_directory(
    path: str,
    ignore: Optional[List[str]] = None,
    respect_git_ignore: Optional[bool] = None,
) -> str:
    """
    Lists files and subdirectories in a directory.
    Args:
        path: The absolute path of the directory to list.
        ignore: A list of glob patterns to ignore.
        respect_git_ignore: Whether to respect .gitignore patterns.
    """
    args = {"path": path, "ignore": ignore, "respect_git_ignore": respect_git_ignore}
    args = {k: v for k, v in args.items() if v is not None}
    result = default_api_module.list_directory(**args)
    return json.dumps(result)


@tool(name="search_file_content", title="Search File Content")
async def search_file_content(
    pattern: str, include: Optional[str] = None, path: Optional[str] = None
) -> str:
    """
    Searches for a regex pattern within file contents.
    Args:
        pattern: The regular expression pattern to search for.
        include: A glob pattern to filter which files are searched.
        path: The directory to search within.
    """
    args = {"pattern": pattern, "include": include, "path": path}
    args = {k: v for k, v in args.items() if v is not None}
    result = default_api_module.search_file_content(**args)
    return json.dumps(result)


@tool(name="glob", title="Find Files (Glob)")
async def glob_tool(
    pattern: str,
    case_sensitive: Optional[bool] = None,
    path: Optional[str] = None,
    respect_git_ignore: Optional[bool] = None,
) -> str:
    """
    Finds files matching a glob pattern.
    Args:
        pattern: The glob pattern to match against (e.g., '**/*.py').
        case_sensitive: Whether the search should be case-sensitive.
        path: The directory to search within.
        respect_git_ignore: Whether to respect .gitignore patterns.
    """
    args = {
        "pattern": pattern,
        "case_sensitive": case_sensitive,
        "path": path,
        "respect_git_ignore": respect_git_ignore,
    }
    args = {k: v for k, v in args.items() if v is not None}
    result = default_api_module.glob(**args)
    return json.dumps(result)


@tool(name="replace", title="Replace Text in File")
async def replace(
    file_path: str,
    new_string: str,
    old_string: str,
    expected_replacements: Optional[float] = None,
) -> str:
    """
    Replaces text within a file.
    Args:
        file_path: The absolute path of the file to modify.
        new_string: The new text to insert.
        old_string: The existing text to be replaced.
        expected_replacements: The number of occurrences to replace.
    """
    args = {
        "file_path": file_path,
        "new_string": new_string,
        "old_string": old_string,
        "expected_replacements": expected_replacements,
    }
    args = {k: v for k, v in args.items() if v is not None}
    result = default_api_module.replace(**args)
    return json.dumps(result)


@tool(name="read_many_files", title="Read Many Files")
async def read_many_files(
    paths: List[str],
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    recursive: Optional[bool] = True,
    respect_git_ignore: Optional[bool] = True,
    useDefaultExcludes: Optional[bool] = True,
) -> str:
    """
    Reads content from multiple files.
    Args:
        paths: A list of glob patterns or file paths.
        exclude: Glob patterns for files/directories to exclude.
        include: Glob patterns to include.
        recursive: Whether to search recursively.
        respect_git_ignore: Whether to respect .gitignore patterns.
        useDefaultExcludes: Whether to apply default exclusion patterns.
    """
    args = {
        "paths": paths,
        "exclude": exclude,
        "include": include,
        "recursive": recursive,
        "respect_git_ignore": respect_git_ignore,
        "useDefaultExcludes": useDefaultExcludes,
    }
    args = {k: v for k, v in args.items() if v is not None}
    result = default_api_module.read_many_files(**args)
    return json.dumps(result)


@tool(name="save_memory", title="Save Memory")
async def save_memory(fact: str) -> str:
    """
    Saves a fact to long-term memory.
    Args:
        fact: The piece of information to remember.
    """
    result = default_api_module.save_memory(fact=fact)
    return json.dumps(result)


# --- List Tools Endpoint ---


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return list(_tools.values())


# --- Main Server Setup ---


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="sse",  # Default to SSE for web-based interaction
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    if transport == "sse":
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            return Response()

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        # Load RAG components before starting the server
        anyio.run(load_rag_components)

        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        # This part is for stdio transport, not typically used for web servers
        from mcp.server.stdio import stdio_server

        async def arun():
            # Load RAG components before starting the server
            await load_rag_components()
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)

    return 0


if __name__ == "__main__":
    main()
