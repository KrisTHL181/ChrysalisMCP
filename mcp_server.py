from typing import Optional
import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
import uvicorn
import inspect
import json
import os
from pathlib import Path
import mimetypes
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

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
        rag_vectorstore = FAISS.load_local(FAISS_INDEX_PATH, rag_embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS vector store loaded.")

        # Load all chunks for BM25
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            all_chunks_data = json.load(f)
        all_chunks = [d['page_content'] for d in all_chunks_data]
        tokenized_corpus = [doc.split(" ") for doc in all_chunks]
        bm25_retriever = BM25Okapi(tokenized_corpus)
        logger.info("BM25 retriever initialized.")

        # Load global memory content
        global_memory_file = os.path.expanduser("~/.ChrysalisMCP/global_memory.md")
        if os.path.exists(global_memory_file):
            with open(global_memory_file, 'r', encoding='utf-8') as f:
                global_memory_content = f.read()
            logger.info("Global memory loaded.")
        else:
            logger.warning("Global memory file not found. It will be created on first update.")

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
ALLOW_ABSOLUTE_PATHS = False # Will be set by CLI option

# Import the default_api which provides access to the CLI tools
import default_api as default_api_module

# Helper function to generate JSON Schema from function signature
def get_json_schema_for_function(func):
    signature = inspect.signature(func)
    properties = {}
    required = []

    type_mapping = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
        None: "null"
    }

    for name, param in signature.parameters.items():
        if name == 'self':
            continue

        param_type = "string"  # Default to string
        is_optional = False
        items_type = "string" # Default for array items

        if hasattr(param.annotation, '__origin__'):
            if param.annotation.__origin__ is Optional:
                is_optional = True
                inner_type = param.annotation.__args__[0]
                if hasattr(inner_type, '__origin__') and inner_type.__origin__ is list:
                    param_type = "array"
                    if inner_type.__args__:
                        items_type = type_mapping.get(inner_type.__args__[0], "string")
                else:
                    param_type = type_mapping.get(inner_type, "string")
            elif param.annotation.__origin__ is list:
                param_type = "array"
                if param.annotation.__args__:
                    items_type = type_mapping.get(param.annotation.__args__[0], "string")
            else:
                param_type = type_mapping.get(param.annotation, "string")
        elif param.annotation is not inspect.Parameter.empty:
            param_type = type_mapping.get(param.annotation, "string")
        elif param.default is not inspect.Parameter.empty:
            param_type = type_mapping.get(type(param.default), "string")

        if param_type == "array":
            properties[name] = {"type": "array", "items": {"type": items_type}}
        else:
            properties[name] = {"type": param_type, "description": f"Parameter {name}"}

        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }

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
                required=True
            )
        ],
    ),
    "explain-code": types.Prompt(
        name="explain-code",
        description="Explain how a given code snippet works.",
        arguments=[
            types.PromptArgument(
                name="code",
                description="The code snippet to explain.",
                required=True
            ),
            types.PromptArgument(
                name="language",
                description="The programming language of the code (e.g., python, javascript).",
                required=False
            )
        ],
    ),
    "summarize-text": types.Prompt(
        name="summarize-text",
        description="Summarize a given text.",
        arguments=[
            types.PromptArgument(
                name="text",
                description="The text to summarize.",
                required=True
            ),
            types.PromptArgument(
                name="length",
                description="Desired length of the summary (e.g., short, medium, long).",
                required=False
            )
        ],
    ),
    "generate-code": types.Prompt(
        name="generate-code",
        description="Generate code based on a description.",
        arguments=[
            types.PromptArgument(
                name="description",
                description="Description of the code to generate.",
                required=True
            ),
            types.PromptArgument(
                name="language",
                description="The programming language for the generated code.",
                required=False
            )
        ],
    ),
    "ask-rag": types.Prompt(
        name="ask-rag",
        description="Ask a question that can be answered by retrieving information from the local knowledge base.",
        arguments=[
            types.PromptArgument(
                name="query",
                description="The question to ask.",
                required=True
            )
        ],
    ),
    "generate-clue": types.Prompt(
        name="generate-clue",
        description="Generate a retrieval clue based on a query and global memory.",
        arguments=[
            types.PromptArgument(
                name="query",
                description="The original user query.",
                required=True
            ),
            types.PromptArgument(
                name="global_memory",
                description="Summary of the global knowledge base.",
                required=False
            )
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
                        text=f"Generate a concise and descriptive Git commit message for the following changes:\n\n{changes}"
                    )
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
                        text=f"Explain how this {language} code works:\n\n```\n{code}\n```"
                    )
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
                        text=f"Summarize the following text. Desired length: {length}.\n\n{text}"
                    )
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
                        text=f"Generate {language} code for the following description:\n\n{description}"
                    )
                )
            ]
        )

    if name == "ask-rag":
        query = arguments.get("query", "") if arguments else ""
        clue_query = arguments.get("clue_query", "") if arguments else "" # Add clue_query argument

        if not query:
            raise ValueError("Missing required argument 'query' for ask-rag")

        if not clue_query: # If clue_query is not provided, generate a clue
            clue_prompt_text = f"""Based on the following original query and global memory summary, generate a more detailed and comprehensive retrieval query (clue) that would help find relevant information. The clue should be a single, concise query.

Original Query: {query}
Global Memory Summary: {global_memory_content}

Generated Clue:"""
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=clue_prompt_text
                        )
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
                            text="RAG knowledge base is not available. Please inform the developer."
                        )
                    )
                ]
            )

        logger.info(f"Performing RAG search for query: {query} with clue: {clue_query}") # Log clue_query
        
        # Semantic search (FAISS)
        faiss_docs = rag_vectorstore.similarity_search(clue_query, k=5) # Use clue_query for search
        faiss_context = "\n\n".join([doc.page_content for doc in faiss_docs])
        logger.debug(f"Retrieved FAISS context:\n{faiss_context}")

        # Keyword search (BM25)
        tokenized_query = clue_query.split(" ") # Use clue_query for BM25
        bm25_scores = bm25_retriever.get_scores(tokenized_query)
        # Get top 5 BM25 documents based on scores
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
        bm25_docs = [all_chunks[i] for i in top_bm25_indices]
        bm25_context = "\n\n".join(bm25_docs)
        logger.debug(f"Retrieved BM25 context:\n{bm25_context}")

        # Combine contexts (simple concatenation for now, can be refined)
        combined_context = f"Semantic Search Results:\n{faiss_context}\n\nKeyword Search Results:\n{bm25_context}"

        # Construct the prompt with retrieved context
        rag_prompt = f"""Based on the following information, answer the question. If the information does not contain the answer, state that you don't know.

Global Memory Summary:
{global_memory_content}

Context:
{combined_context}

Question: {query}
Answer:"""

        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=rag_prompt
                    )
                )
            ]
        )

    if name == "generate-clue":
        query = arguments.get("query", "") if arguments else ""
        global_memory = arguments.get("global_memory", "") if arguments else ""
        
        clue_prompt_text = f"""Based on the following original query and global memory summary, generate a more detailed and comprehensive retrieval query (clue) that would help find relevant information. The clue should be a single, concise query.

Original Query: {query}
Global Memory Summary: {global_memory}

Generated Clue:"""

        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=clue_prompt_text
                    )
                )
            ]
        )

    raise ValueError("Prompt implementation not found")

# Helper for path validation
def validate_path_for_mcp_tool(input_path: str, is_write_operation: bool = False) -> Path:
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
                raise ValueError(f"Access denied: Path '{input_path}' is outside allowed directories (current working directory or mcp_resources).")


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    resources = []
    # Ensure the base directory exists
    RESOURCE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Dynamically list files in the resource base directory
    for file_path in RESOURCE_BASE_DIR.rglob("*"): # rglob for recursive search
        if file_path.is_file():
            relative_path = file_path.relative_to(RESOURCE_BASE_DIR)
            uri = f"file:///{relative_path.as_posix()}" # Use posix path for URI consistency
            name = relative_path.name
            mime_type, _ = mimetypes.guess_type(file_path.name)
            if mime_type is None:
                mime_type = "application/octet-stream" # Default to generic binary if type cannot be guessed

            resources.append(
                types.Resource(
                    uri=uri,
                    name=name,
                    mimeType=mime_type,
                    description=f"Dynamically discovered resource: {relative_path}"
                )
            )
    return resources

@app.read_resource()
async def read_resource(uri: str) -> str:
    # Extract the relative path from the URI
    if not uri.startswith("file:///"):
        raise ValueError(f"Unsupported resource URI scheme: {uri}")

    relative_path_str = uri[len("file:///"):]
    file_path = RESOURCE_BASE_DIR / relative_path_str

    if not file_path.is_file() or not file_path.exists():
        raise ValueError(f"Resource not found: {uri}")

    # Ensure the file is within the allowed base directory
    try:
        file_path.relative_to(RESOURCE_BASE_DIR)
    except ValueError:
        raise ValueError(f"Access denied: {uri} is outside the allowed resource directory.")

    # Use the default_api_module.read_file to read the content
    # Note: default_api_module.read_file currently returns text.
    # For binary files, you would need to handle base64 encoding.
    result = default_api_module.read_file(absolute_path=str(file_path))
    if "error" in result:
        raise ValueError(f"Failed to read resource {uri}: {result["error"]}")
    return result["output"]

# --- Tool Implementations (wrapping default_api_module functions) ---

import mcp_calculator

@app.call_tool()
async def calculate(name: str, arguments: dict) -> str:
    if name != "calculate":
        raise ValueError(f"Unknown tool: {name}")
    expression = arguments.get("expression")
    if not expression:
        raise ValueError("Missing required argument 'expression' for calculate")
    try:
        result = mcp_calculator.calculate(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

@app.call_tool()
async def google_search(name: str, arguments: dict) -> str:
    if name != "google_search":
        raise ValueError(f"Unknown tool: {name}")
    query = arguments.get("query")
    if not query:
        raise ValueError("Missing required argument 'query' for google_search")
    result = default_api_module.google_web_search(query=query)
    return json.dumps(result)

@app.call_tool()
async def web_fetch(name: str, arguments: dict) -> str:
    if name != "web_fetch":
        raise ValueError(f"Unknown tool: {name}")
    prompt = arguments.get("prompt")
    if not prompt:
        raise ValueError("Missing required argument 'prompt' for web_fetch")
    result = default_api_module.web_fetch(prompt=prompt)
    return json.dumps(result)

@app.call_tool()
async def shell(name: str, arguments: dict) -> str:
    if name != "shell":
        raise ValueError(f"Unknown tool: {name}")
    command = arguments.get("command")
    if not command:
        raise ValueError("Missing required argument 'command' for shell")
    result = default_api_module.run_shell_command(**arguments)
    return json.dumps(result)

@app.call_tool()
async def read_file(name: str, arguments: dict) -> str:
    if name != "read_file":
        raise ValueError(f"Unknown tool: {name}")
    absolute_path = arguments.get("absolute_path")
    if not absolute_path:
        raise ValueError("Missing required argument 'absolute_path' for read_file")

    # Validate path
    validated_path = validate_path_for_mcp_tool(absolute_path)
    arguments["absolute_path"] = str(validated_path)

    result = default_api_module.read_file(**arguments)
    return json.dumps(result)

@app.call_tool()
async def write_file(name: str, arguments: dict) -> str:
    if name != "write_file":
        raise ValueError(f"Unknown tool: {name}")
    file_path = arguments.get("file_path")
    content = arguments.get("content")
    if not file_path or not content:
        raise ValueError("Missing required arguments 'file_path' or 'content' for write_file")

    # Validate path for write operation
    validated_path = validate_path_for_mcp_tool(file_path, is_write_operation=True)
    arguments["file_path"] = str(validated_path)

    result = default_api_module.write_file(**arguments)
    return json.dumps(result)

@app.call_tool()
async def list_directory(name: str, arguments: dict) -> str:
    if name != "list_directory":
        raise ValueError(f"Unknown tool: {name}")
    path = arguments.get("path")
    if not path:
        raise ValueError("Missing required argument 'path' for list_directory")

    # Validate path
    validated_path = validate_path_for_mcp_tool(path)
    arguments["path"] = str(validated_path)

    result = default_api_module.list_directory(**arguments)
    return json.dumps(result)

@app.call_tool()
async def search_file_content(name: str, arguments: dict) -> str:
    if name != "search_file_content":
        raise ValueError(f"Unknown tool: {name}")
    pattern = arguments.get("pattern")
    if not pattern:
        raise ValueError("Missing required argument 'pattern' for search_file_content")

    if "path" in arguments and arguments["path"] is not None:
        validated_path = validate_path_for_mcp_tool(arguments["path"])
        arguments["path"] = str(validated_path)

    result = default_api_module.search_file_content(**arguments)
    return json.dumps(result)

@app.call_tool()
async def glob_tool(name: str, arguments: dict) -> str:
    if name != "glob":
        raise ValueError(f"Unknown tool: {name}")
    pattern = arguments.get("pattern")
    if not pattern:
        raise ValueError("Missing required argument 'pattern' for glob")

    if "path" in arguments and arguments["path"] is not None:
        validated_path = validate_path_for_mcp_tool(arguments["path"])
        arguments["path"] = str(validated_path)

    result = default_api_module.glob(**arguments)
    return json.dumps(result)

@app.call_tool()
async def replace(name: str, arguments: dict) -> str:
    if name != "replace":
        raise ValueError(f"Unknown tool: {name}")
    file_path = arguments.get("file_path")
    new_string = arguments.get("new_string")
    old_string = arguments.get("old_string")
    if not file_path or not new_string or not old_string:
        raise ValueError("Missing required arguments 'file_path', 'new_string', or 'old_string' for replace")

    # Validate path for write operation
    validated_path = validate_path_for_mcp_tool(file_path, is_write_operation=True)
    arguments["file_path"] = str(validated_path)

    result = default_api_module.replace(**arguments)
    return json.dumps(result)

@app.call_tool()
async def read_many_files(name: str, arguments: dict) -> str:
    if name != "read_many_files":
        raise ValueError(f"Unknown tool: {name}")
    paths = arguments.get("paths")
    if not paths:
        raise ValueError("Missing required argument 'paths' for read_many_files")

    # Validate each path in the list
    validated_paths = []
    for p in paths:
        validated_paths.append(str(validate_path_for_mcp_tool(p)))
    arguments["paths"] = validated_paths

    result = default_api_module.read_many_files(**arguments)
    return json.dumps(result)

@app.call_tool()
async def save_memory(name: str, arguments: dict) -> str:
    if name != "save_memory":
        raise ValueError(f"Unknown tool: {name}")
    fact = arguments.get("fact")
    if not fact:
        raise ValueError("Missing required argument 'fact' for save_memory")
    result = default_api_module.save_memory(**arguments)
    return json.dumps(result)

# --- List Tools Endpoint ---

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = [
        types.Tool(
            name="google_search",
            title="Google Search",
            description="Performs a web search using Google Search.",
            inputSchema=get_json_schema_for_function(default_api_module.google_web_search)
        ),
        types.Tool(
            name="web_fetch",
            title="Web Fetch",
            description="Fetches content from URLs specified in the prompt.",
            inputSchema=get_json_schema_for_function(default_api_module.web_fetch)
        ),
        types.Tool(
            name="shell",
            title="Shell Command",
            description="Executes a shell command.",
            inputSchema=get_json_schema_for_function(default_api_module.run_shell_command)
        ),
        types.Tool(
            name="read_file",
            title="Read File",
            description="Reads content from a specified file.",
            inputSchema=get_json_schema_for_function(default_api_module.read_file)
        ),
        types.Tool(
            name="write_file",
            title="Write File",
            description="Writes content to a specified file.",
            inputSchema=get_json_schema_for_function(default_api_module.write_file)
        ),
        types.Tool(
            name="list_directory",
            title="List Directory",
            description="Lists the names of files and subdirectories within a specified directory path.",
            inputSchema=get_json_schema_for_function(default_api_module.list_directory)
        ),
        types.Tool(
            name="search_file_content",
            title="Search File Content",
            description="Searches for a regular expression pattern within the content of files.",
            inputSchema=get_json_schema_for_function(default_api_module.search_file_content)
        ),
        types.Tool(
            name="glob",
            title="Find Files (Glob)",
            description="Efficiently finds files matching specific glob patterns.",
            inputSchema=get_json_schema_for_function(default_api_module.glob)
        ),
        types.Tool(
            name="replace",
            title="Replace Text in File",
            description="Replaces text within a file.",
            inputSchema=get_json_schema_for_function(default_api_module.replace)
        ),
        types.Tool(
            name="read_many_files",
            title="Read Many Files",
            description="Reads content from multiple files specified by paths or glob patterns.",
            inputSchema=get_json_schema_for_function(default_api_module.read_many_files)
        ),
        types.Tool(
            name="save_memory",
            title="Save Memory",
            description="Saves a specific piece of information or fact to your long-term memory.",
            inputSchema=get_json_schema_for_function(default_api_module.save_memory)
        ),
        types.Tool(
            name="calculate",
            title="Calculator",
            description="Evaluates a mathematical expression.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to evaluate."}
                },
                "required": ["expression"]
            }
        ),
    ]
    return tools

# --- Main Server Setup ---

@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="sse", # Default to SSE for web-based interaction
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    if transport == "sse":
        sse = SseServerTransport("/messages/")

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=sse.handle_sse, methods=["GET"]),
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