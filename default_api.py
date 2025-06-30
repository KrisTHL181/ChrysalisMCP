import os
import datetime
import subprocess
import requests
import re
import glob
import base64
import mimetypes
from typing import List, Dict, Optional
from loguru import logger
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# --- API Key Placeholders ---
# IMPORTANT: Replace these with your actual API keys
GOOGLE_SEARCH_API_KEY = "YOUR_GOOGLE_SEARCH_API_KEY"
GOOGLE_SEARCH_CX = "YOUR_GOOGLE_SEARCH_CX" # Custom Search Engine ID

# --- Tool Implementations ---

def google_web_search(query: str) -> Dict:
    """Performs a web search using Google Search via.
    
    Args:
        query: The search query to find information on the web.
    """
    logger.debug(f"Performing Google Web Search for: {query}")
    if GOOGLE_SEARCH_API_KEY == "YOUR_GOOGLE_SEARCH_API_KEY":
        logger.warning("Google Search API Key not configured!")
        return {
            "search_results": [
                {"title": "ALERT", "link": "MCP Server", "snippet": "Google Search API is not configured. Please inform the developer to configure it!"},
            ]
        }
    
    # Real API call (requires Google Custom Search API enabled and a CX ID)
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_API_KEY}&cx={GOOGLE_SEARCH_CX}&q={query}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {"search_results": data.get("items", [])}
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Search API call failed: {e}")
        return {"error": f"Google Search API call failed: {e}"}

def web_fetch(prompt: str) -> Dict:
    """Processes content from URL(s) embedded in a prompt.
    
    Args:
        prompt: A comprehensive prompt that includes the URL(s) (up to 20) to fetch and specific instructions on how to process their content.
    """
    logger.debug(f"Performing Web Fetch for: {prompt}")
    urls = re.findall(r'https?://[^\s]+', prompt)
    results = []
    if not urls:
        return {"error": "No URLs found in the prompt."}

    for url in urls:
        try:
            response = requests.get(url, timeout=10) # Add a timeout for requests
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            results.append({"url": url, "content": response.text})
        except requests.exceptions.RequestException as e:
            results.append({"url": url, "error": str(e)})
    return {"fetched_content": results}

def run_shell_command(command: str, description: Optional[str] = None, directory: Optional[str] = None) -> Dict:
    """Executes a given shell command.
    
    Args:
        command: Exact bash command to execute as `bash -c <command>`.
        description: Brief description of the command for the user.
        directory: Directory to run the command in.
    """
    logger.debug(f"Running shell command: {command} (Description: {description}, Directory: {directory})")
    try:
        # Use subprocess to run the command
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=directory if directory else None,
            check=True # Raise an exception for non-zero exit codes
        )
        return {
            "Command": command,
            "Directory": directory if directory else "(root)",
            "Stdout": process.stdout.strip() if process.stdout else "(empty)",
            "Stderr": process.stderr.strip() if process.stderr else "(empty)",
            "Error": "(none)",
            "Exit Code": process.returncode,
            "Signal": "(none)",
            "Background PIDs": "(none)",
            "Process Group PGID": "(none)"
        }
    except subprocess.CalledProcessError as e:
        return {
            "Command": command,
            "Directory": directory if directory else "(root)",
            "Stdout": e.stdout.strip() if e.stdout else "(empty)",
            "Stderr": e.stderr.strip() if e.stderr else "(empty)",
            "Error": f"Command failed with exit code {e.returncode}",
            "Exit Code": e.returncode,
            "Signal": "(none)",
            "Background PIDs": "(none)",
            "Process Group PGID": "(none)"
        }
    except Exception as e:
        return {"error": f"Failed to execute shell command: {e}"}

def read_file(absolute_path: str, limit: Optional[float] = None, offset: Optional[float] = None) -> Dict:
    """Reads and returns the content of a specified file, handling text, images, and PDFs.
    
    Args:
        absolute_path: The absolute path to the file to read.
        limit: For text files, maximum number of lines to read.
        offset: For text files, the 0-based line number to start reading from.
    """

    logger.debug(f"Reading file: {absolute_path}")
    try:
        mime_type, _ = mimetypes.guess_type(absolute_path)
        is_text = mime_type and mime_type.startswith('text/')

        if is_text:
            with open(absolute_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if limit is not None and offset is not None:
                    content = "".join(lines[int(offset):int(offset + limit)])
                else:
                    content = "".join(lines)
            return {"output": content, "encoding": "utf-8"}
        else:
            # For non-text files, read as binary and encode in base64
            with open(absolute_path, 'rb') as f:
                binary_content = f.read()
            encoded_content = base64.b64encode(binary_content).decode('utf-8')
            return {"output": encoded_content, "encoding": "base64", "mime_type": mime_type}
            
    except FileNotFoundError:
        return {"error": f"File not found: {absolute_path}"}
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

def write_file(content: str, file_path: str) -> Dict:
    """Writes content to a specified file.
    
    Args:
        content: The content to write to the file.
        file_path: The absolute path to the file to write to.
    """
    logger.debug(f"Writing to file: {file_path}")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"output": f"Successfully wrote to {file_path}"}
    except Exception as e:
        return {"error": f"Failed to write file: {e}"}

def list_directory(path: str, ignore: Optional[List[str]] = None, respect_git_ignore: Optional[bool] = None) -> Dict:
    """Lists the names of files and subdirectories directly within a specified directory path.
    
    Args:
        path: The absolute path to the directory to list.
        ignore: List of glob patterns to ignore.
        respect_git_ignore: Whether to respect .gitignore patterns.
    """
    logger.debug(f"Listing directory: {path}")
    try:
        entries = os.listdir(path)
        
        ignore_patterns = ignore.copy() if ignore else []

        if respect_git_ignore:
            gitignore_path = os.path.join(path, '.gitignore')
            if os.path.exists(gitignore_path):
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            ignore_patterns.append(line)

        if ignore_patterns:
            filtered_entries = []
            for entry in entries:
                if not any(glob.fnmatch.fnmatch(entry, p) for p in ignore_patterns):
                    filtered_entries.append(entry)
            entries = filtered_entries

        return {"output": f"Directory listing for {path}:\n" + "\n".join(entries)}
    except FileNotFoundError:
        return {"error": f"Directory not found: {path}"}
    except Exception as e:
        return {"error": f"Failed to list directory: {e}"}

def search_file_content(pattern: str, include: Optional[str] = None, path: Optional[str] = None) -> Dict:
    """Searches for a regular expression pattern within the content of files.
    
    Args:
        pattern: The regular expression (regex) pattern to search for.
        include: A glob pattern to filter which files are searched.
        path: The absolute path to the directory to search within.
    """
    logger.debug(f"Searching file content for pattern: {pattern} in path: {path}")
    matches = []
    target_path = path if path else os.getcwd()
    for root, _, files in os.walk(target_path):
        for file in files:
            if include and not glob.fnmatch.fnmatch(file, include):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if re.search(pattern, line):
                            matches.append(f"File: {file_path}, Line {i+1}: {line.strip()}")
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
    return {"output": "\n".join(matches) if matches else "No matches found."}

def glob(pattern: str, case_sensitive: Optional[bool] = None, path: Optional[str] = None, respect_git_ignore: Optional[bool] = None) -> Dict:
    """Efficiently finds files matching specific glob patterns, with gitignore support and sorting.
    
    Args:
        pattern: The glob pattern to match against.
        case_sensitive: Whether the search should be case-sensitive.
        path: The absolute path to the directory to search within.
        respect_git_ignore: Whether to respect .gitignore patterns.
    """
    logger.debug(f"Globbing for pattern: {pattern} in path: {path}")
    target_path = path if path else os.getcwd()
    
    # Get all files matching the pattern
    all_files = [os.path.join(root, file) 
                 for root, _, files in os.walk(target_path) 
                 for file in files if glob.fnmatch.fnmatch(file, pattern)]

    # Filter based on .gitignore if requested
    if respect_git_ignore:
        gitignore_path = os.path.join(target_path, '.gitignore')
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                gitignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            files_to_keep = []
            for file in all_files:
                relative_path = os.path.relpath(file, target_path)
                if not any(glob.fnmatch.fnmatch(relative_path, p) for p in gitignore_patterns):
                    files_to_keep.append(file)
            all_files = files_to_keep

    # Sort files by modification time (newest first)
    try:
        all_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    except FileNotFoundError as e:
        # This can happen if a file is deleted during the sort
        logger.warning(f"File not found during sort: {e}")

    return {"output": "\n".join(all_files)}

def replace(file_path: str, new_string: str, old_string: str, expected_replacements: Optional[float] = None) -> Dict:
    """Replaces text within a file.
    
    Args:
        file_path: The absolute path to the file to modify.
        new_string: The new string to replace the old one with.
        old_string: The string to be replaced.
        expected_replacements: The number of occurrences to replace.
    """
    logger.debug(f"Replacing content in file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if expected_replacements is None:
            # Replace only the first occurrence
            new_content = content.replace(old_string, new_string, 1)
            if new_content == content:
                return {"error": "No occurrence found for single replacement."}
        else:
            # Replace all occurrences
            new_content = content.replace(old_string, new_string)
            actual_replacements = content.count(old_string)
            if actual_replacements != expected_replacements:
                logger.warning(f"Expected {expected_replacements} replacements but found {actual_replacements}.")
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return {"output": f"Successfully replaced content in {file_path}"}
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to replace content: {e}"}

def read_many_files(paths: List[str], exclude: Optional[List[str]] = None, include: Optional[List[str]] = None, recursive: Optional[bool] = True, respect_git_ignore: Optional[bool] = True, useDefaultExcludes: Optional[bool] = True) -> Dict:
    """Reads content from multiple files specified by paths or glob patterns, with advanced filtering.
    
    Args:
        paths: An array of glob patterns or paths.
        exclude: Glob patterns for files/directories to exclude.
        include: Additional glob patterns to include.
        recursive: Whether to search recursively.
        respect_git_ignore: Whether to respect .gitignore patterns.
        useDefaultExcludes: Whether to apply a list of default exclusion patterns.
    """
    logger.debug(f"Reading many files from paths: {paths}")
    all_content = []
    
    # Use the new glob function to get a file list
    all_files = []
    for p in paths:
        glob_result = glob(pattern=p, respect_git_ignore=respect_git_ignore)
        if "output" in glob_result:
            all_files.extend(glob_result["output"].splitlines())

    # Apply exclude and include patterns
    if exclude:
        all_files = [f for f in all_files if not any(glob.fnmatch.fnmatch(os.path.basename(f), pat) for pat in exclude)]
    if include:
        all_files = [f for f in all_files if any(glob.fnmatch.fnmatch(os.path.basename(f), pat) for pat in include)]

    for file_path in all_files:
        if os.path.isfile(file_path):
            try:
                read_result = read_file(file_path)
                if "output" in read_result:
                    all_content.append(f"--- {file_path} ---\n{read_result['output']}")
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                
    return {"output": "\n\n".join(all_content) if all_content else "No files read."}

def save_memory(fact: str) -> Dict:
    """Saves a specific piece of information or fact to your long-term memory.
    
    Args:
        fact: The specific fact or piece of information to remember.
    """
    logger.debug(f"Saving memory: {fact}")
    try:
        # Use a file in the user's home directory to store memories
        memory_dir = os.path.expanduser("~/.ChrysalisMCP")
        os.makedirs(memory_dir, exist_ok=True)
        memory_file = os.path.join(memory_dir, "ChrysalisMCP.md")
        
        # Add a timestamp to the memory
        timestamp = datetime.datetime.now().isoformat()
        
        with open(memory_file, 'a', encoding='utf-8') as f:
            f.write(f"- {fact} (saved at {timestamp})\n")
            
        return {"output": f"Fact '{fact}' saved to {memory_file}."}
    except Exception as e:
        return {"error": f"Failed to save memory: {e}"}

def update_global_memory(summary_model_name: str = "sshleifer/distilbart-cnn-12-6", summary: str = None) -> Dict:
    """Updates the global memory with a summary of the knowledge base, or generates one using a local LLM."""
    logger.debug(f"Updating global memory.")
    try:
        memory_dir = os.path.expanduser("~/.ChrysalisMCP")
        os.makedirs(memory_dir, exist_ok=True)
        global_memory_file = os.path.join(memory_dir, "global_memory.md")

        if summary is None:
            # If no summary is provided, generate one using local LLM
            logger.info(f"Generating global memory summary using local LLM: {summary_model_name}...")
            
            # Load all documents from mcp_resources
            RESOURCES_DIR = os.path.join(os.getcwd(), "mcp_resources")
            documents = []
            
            txt_loader = DirectoryLoader(RESOURCES_DIR, loader_cls=TextLoader, glob="**/*.txt")
            documents.extend(txt_loader.load())
            pdf_loader = DirectoryLoader(RESOURCES_DIR, loader_cls=PyPDFLoader, glob="**/*.pdf")
            documents.extend(pdf_loader.load())
            docx_loader = DirectoryLoader(RESOURCES_DIR, loader_cls=Docx2txtLoader, glob="**/*.docx")
            documents.extend(docx_loader.load())
            pptx_loader = DirectoryLoader(RESOURCES_DIR, loader_cls=UnstructuredPowerPointLoader, glob="**/*.pptx")
            documents.extend(pptx_loader.load())

            if not documents:
                logger.warning("No documents found in mcp_resources to summarize for global memory.")
                summary = "No documents available for global memory."
            else:
                # Combine all document content
                full_content = "\n\n".join([doc.page_content for doc in documents])

                # Use local LLM to summarize
                summarizer = pipeline("summarization", model=summary_model_name)
                
                # Split content into chunks if too long for LLM
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                texts = text_splitter.split_text(full_content)

                summaries = []
                for i, text_chunk in enumerate(texts):
                    logger.info(f"Summarizing chunk {i+1}/{len(texts)} for global memory...")
                    # The summarizer pipeline expects a list of strings
                    summary_result = summarizer(text_chunk, max_length=150, min_length=30, do_sample=False)
                    summaries.append(summary_result[0]['summary_text'])
                
                summary = "\n\n".join(summaries)
                logger.info("Global memory summary generated.")

        with open(global_memory_file, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        return {"output": f"Global memory updated in {global_memory_file}."}
    except Exception as e:
        return {"error": f"Failed to update global memory: {e}"}

# Create a default_api object for consistency with how it was used before
class DefaultApiWrapper:
    def __init__(self):
        self.google_web_search = google_web_search
        self.web_fetch = web_fetch
        self.run_shell_command = run_shell_command
        self.read_file = read_file
        self.write_file = write_file
        self.list_directory = list_directory
        self.search_file_content = search_file_content
        self.glob = glob
        self.replace = replace
        self.read_many_files = read_many_files
        self.save_memory = save_memory
        self.update_global_memory = update_global_memory

default_api = DefaultApiWrapper()