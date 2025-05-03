import os
import shutil
import subprocess
import typer
from typing import List, Optional, Dict
from tree_sitter import Language, Parser
import tree_sitter_languages
import litellm
import json
import tiktoken
from loguru import logger
import textwrap
from markitdown import MarkItDown
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from complexity.gitgit.initialize_litellm_cache import initialize_litellm_cache
from complexity.gitgit.json_utils import clean_json_string, json_to_markdown

load_dotenv()
initialize_litellm_cache()

# Supported languages and their extensions
LANGUAGE_EXTENSIONS = {
    'python': ['py'],
    'javascript': ['js'],
    'typescript': ['ts'],
    'java': ['java'],
    'cpp': ['cpp', 'c'],
    'go': ['go'],
    'ruby': ['rb']
}

app = typer.Typer(help="A CLI utility for sparse cloning, summarizing, and LLM-based documentation of GitHub repositories.")

def count_tokens_with_tiktoken(text, model="gemini-2.5-pro-preview-03-25"):
    try:
        import tiktoken
        openai_models = {
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k",
            "gpt-4o", "gpt-4-turbo"
        }
        if any(model.startswith(m) for m in openai_models):
            encoding = tiktoken.encoding_for_model(model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return int(len(text) / 4)
    except Exception:
        return int(len(text) / 4)

def extract_code_metadata(file_path: str, language: str) -> Dict:
    """
    Extract function names, parameters, and docstrings from a code file using Tree-sitter.
    
    Args:
        file_path (str): Path to the code file.
        language (str): Language identifier (e.g., 'python', 'javascript').
    
    Returns:
        Dict: Metadata containing function definitions.
    """
    try:
        parser = tree_sitter_languages.get_parser(language)
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        metadata = {'language': language, 'functions': []}

        # Language-specific queries
        queries = {
            'python': """
                (function_definition
                    name: (identifier) @func_name
                    parameters: (parameters) @params
                ) @function
            """,
            'javascript': """
                (function_declaration
                    name: (identifier) @func_name
                    parameters: (formal_parameters) @params
                ) @function
            """,
            'typescript': """
                (function_declaration
                    name: (identifier) @func_name
                    parameters: (formal_parameters) @params
                ) @function
            """,
            'java': """
                (method_declaration
                    name: (identifier) @func_name
                    parameters: (formal_parameters) @params
                ) @function
            """,
            'cpp': """
                (function_declarator
                    declarator: (identifier) @func_name
                    parameters: (parameter_list) @params
                ) @function
            """,
            'go': """
                (function_declaration
                    name: (identifier) @func_name
                    parameters: (parameter_list) @params
                ) @function
            """,
            'ruby': """
                (method
                    name: (identifier) @func_name
                    parameters: (method_parameters) @params
                ) @function
            """
        }

        def get_docstring(node, lang):
            if lang in ['python', 'javascript', 'typescript', 'ruby']:
                for child in node.children:
                    if child.type in ['expression_statement', 'comment']:
                        for grandchild in child.children:
                            if grandchild.type in ['string', 'comment']:
                                return grandchild.text.decode('utf8').strip('\'"//')
            return ''

        query_str = queries.get(language, '')
        if not query_str:
            logger.warning(f"No query defined for language: {language}")
            return metadata

        language_obj = tree_sitter_languages.get_language(language)
        query = language_obj.query(query_str)
        captures = query.captures(root_node)

        for capture, name in captures:
            if name == 'function':
                func_node = capture
                func_name_node = func_node.child_by_field_name('name')
                params_node = func_node.child_by_field_name('parameters')
                if func_name_node and params_node:
                    func_name = func_name_node.text.decode('utf8')
                    params = [param.text.decode('utf8') for param in params_node.children if param.type in ('identifier', 'parameter', 'formal_parameter', 'default_parameter')]
                    docstring = get_docstring(func_node, language)
                    metadata['functions'].append({
                        'name': func_name,
                        'parameters': params,
                        'docstring': docstring
                    })
        
        return metadata
    except Exception as e:
        logger.warning(f"Failed to extract metadata from {file_path}: {e}")
        return {'language': language, 'functions': []}

def interactive_file_selection(repo_url: str) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Placeholder for interactive file/directory selection (browser-based or VS Code-integrated).
    
    Args:
        repo_url (str): GitHub repository URL.
    
    Returns:
        tuple: (files, dirs) selected by the user (currently returns None, None).
    
    Note:
        Future implementation will use a Flask/FastAPI web app or VS Code extension
        to allow users to browse and select files/directories via GitHub API or repository tree view.
    """
    logger.info("Interactive file selection is not yet implemented.")
    return None, None

def sparse_clone(repo_url: str, extensions: List[str], clone_dir: str, files: Optional[List[str]] = None, dirs: Optional[List[str]] = None):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    os.makedirs(clone_dir, exist_ok=True)

    subprocess.run(['git', 'init'], cwd=clone_dir, check=True)
    subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=clone_dir, check=True)
    subprocess.run(['git', 'config', 'core.sparseCheckout', 'true'], cwd=clone_dir, check=True)

    sparse_patterns = []
    if files or dirs:
        if files:
            sparse_patterns.extend([f"{f}" for f in files])
        if dirs:
            sparse_patterns.extend([f"{d.rstrip('/')}/**/*" for d in dirs])
    else:
        for ext in extensions:
            sparse_patterns.append(f'**/*.{ext}')
            sparse_patterns.append(f'/*.{ext}')
    
    sparse_file = os.path.join(clone_dir, '.git', 'info', 'sparse-checkout')
    with open(sparse_file, 'w') as f:
        f.write('\n'.join(sparse_patterns) + '\n')

    subprocess.run(['git', 'pull', '--depth=1', 'origin', 'HEAD'], cwd=clone_dir, check=True)

def save_to_root(root_dir: str, filename: str, content: str):
    with open(os.path.join(root_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)

def build_tree(root_dir):
    tree_lines = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)
        indent = "" if rel_dir == "." else "    " * rel_dir.count(os.sep)
        tree_lines.append(f"{indent}{os.path.basename(dirpath) if rel_dir != '.' else '.'}/")
        for filename in sorted(filenames):
            tree_lines.append(f"{indent}    {filename}")
    return "\n".join(tree_lines)

def concat_and_summarize(root_dir, extensions, files: Optional[List[str]] = None, dirs: Optional[List[str]] = None, code_metadata: bool = False):
    digest_parts = []
    file_count = 0
    total_bytes = 0
    files_list = []
    requested_paths = set(files or []) | {os.path.join(d, f) for d in (dirs or []) for f in os.listdir(os.path.join(root_dir, d)) if os.path.isfile(os.path.join(root_dir, d, f))}

    def get_language(ext):
        for lang, exts in LANGUAGE_EXTENSIONS.items():
            if ext.lower() in exts:
                return lang
        return None

    if files or dirs:
        for path in requested_paths:
            full_path = os.path.join(root_dir, path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                file_count += 1
                relpath = os.path.relpath(full_path, root_dir)
                files_list.append(relpath)
                with open(full_path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                total_bytes += len(content.encode("utf-8"))
                digest_parts.append("="*48)
                digest_parts.append(f"File: {relpath}")
                digest_parts.append("="*48)
                digest_parts.append(content)
                digest_parts.append("")
                if code_metadata:
                    ext = os.path.splitext(relpath)[1].lstrip('.')
                    language = get_language(ext)
                    if language:
                        metadata = extract_code_metadata(full_path, language)
                        digest_parts.append(f"Metadata (JSON):")
                        digest_parts.append(json.dumps(metadata, indent=2))
                        digest_parts.append("")
            else:
                logger.warning(f"Requested path not found: {path}")
    else:
        for ext in extensions:
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in sorted(filenames):
                    if filename.lower().endswith(f".{ext.lower()}"):
                        file_count += 1
                        path = os.path.join(dirpath, filename)
                        relpath = os.path.relpath(path, root_dir)
                        files_list.append(relpath)
                        with open(path, encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        total_bytes += len(content.encode("utf-8"))
                        digest_parts.append("="*48)
                        digest_parts.append(f"File: {relpath}")
                        digest_parts.append("="*48)
                        digest_parts.append(content)
                        digest_parts.append("")
                        if code_metadata:
                            language = get_language(ext)
                            if language:
                                metadata = extract_code_metadata(path, language)
                                digest_parts.append(f"Metadata (JSON):")
                                digest_parts.append(json.dumps(metadata, indent=2))
                                digest_parts.append("")

    digest = "\n".join(digest_parts)
    tree = build_tree(root_dir)

    estimated_tokens = count_tokens_with_tiktoken(digest, model="gemini-2.5-pro-preview-03-25")
    summary = (
        f"Directory: {root_dir}\n"
        f"Files analyzed: {file_count}\n"
        f"Total bytes: {total_bytes}\n"
        f"Estimated tokens: {estimated_tokens}\n"
        f"Files included:\n" + "\n".join(files_list)
    )

    return summary, tree, digest

def debug_print_files(clone_dir, extensions, files: Optional[List[str]] = None, dirs: Optional[List[str]] = None):
    found = []
    if files or dirs:
        requested_paths = set(files or []) | {os.path.join(d, f) for d in (dirs or []) for f in os.listdir(os.path.join(clone_dir, d)) if os.path.isfile(os.path.join(clone_dir, d, f))}
        for path in requested_paths:
            full_path = os.path.join(clone_dir, path)
            if os.path.exists(full_path):
                found.append(os.path.relpath(full_path, clone_dir))
    else:
        for ext in extensions:
            for root, _, files in os.walk(clone_dir):
                for file in files:
                    if file.lower().endswith(f'.{ext.lower()}'):
                        found.append(os.path.relpath(os.path.join(root, file), clone_dir))
    return found

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def llm_summarize(
    digest_path: str,
    summary_path: str,
    model: str = "gemini-2.5-pro-preview-03-25",
    google_vertex_project: str = "gen-lang-client-0870473940",
    google_vertex_location: str = "us-central1",
    output_format: str = "markdown"
):
    import tempfile
    from markitdown import MarkItDown

    with open(digest_path, "r", encoding="utf-8") as f:
        digest_text = f.read()

    system_prompt = (
        "You are an expert technical documentation summarizer. "
        "You are also a JSON validator. You will only output valid JSON. "
        "When summarizing, incorporate any code metadata (e.g., function names, parameters, docstrings) provided."
    )
    user_prompt = textwrap.dedent(f"""
        Given the following repository content, including code metadata where available, return a JSON object with:
        - summary: A concise, clear summary of the repository for technical users, highlighting key functions if metadata is present.
        - table_of_contents: An ordered list of file or section names that represent the structure of the repository.
        - key_sections: (optional) A list of the most important files or sections, with a 1-2 sentence description for each.

        Format your response as valid JSON. Only output the JSON.

        Repository content:
        {digest_text}
    """)

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            google_vertex_project=google_vertex_project,
            google_vertex_location=google_vertex_location,
        )
        if hasattr(response, "choices"):
            content = response.choices[0].message.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)

        content = clean_json_string(content, return_dict=True)
        try:
            parsed = RepoSummary.model_validate(content)
            summary_json = json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse or validate LLM output: {e}")
            summary_json = json.dumps({"error": "Failed to parse or validate LLM output", "raw": content})

        if output_format == "json":
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_json)
            logger.info(f"LLM summary saved to {summary_path} (JSON format)")
        else:
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp_json:
                tmp_json.write(summary_json)
                tmp_json_path = tmp_json.name

            try:
                markdown_content = json_to_markdown(parsed.model_dump())
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                    logger.info(f"LLM summary saved to {summary_path} (Markdown format)")
            finally:
                os.remove(tmp_json_path)

    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"error": str(e)}))
        raise

@app.command()
def analyze(
    repo_url: str = typer.Argument(
        ...,
        help="GitHub repository URL to analyze (e.g. https://github.com/arangodb/python-arango)."
    ),
    extensions: str = typer.Option(
        "md,rst", "--exts", "-e",
        help="Comma-separated list of file extensions to include (e.g. py,md,txt). Ignored if --files or --dirs is provided."
    ),
    files: Optional[str] = typer.Option(
        None, "--files",
        help="Comma-separated list of specific file paths to include (e.g. README.md,src/main.py)."
    ),
    dirs: Optional[str] = typer.Option(
        None, "--dirs",
        help="Comma-separated list of directories to include (e.g. docs/,src/)."
    ),
    summary: bool = typer.Option(
        False, "--summary",
        help="If set, generate an LLM-based summary of the repository digest."
    ),
    code_metadata: bool = typer.Option(
        False, "--code-metadata",
        help="If set, extract function metadata (names, parameters, docstrings) from code files."
    ),
    llm_model: Optional[str] = typer.Option(
        "gemini-2.5-pro-preview-03-25", "--llm-model",
        help="LLM model name for LiteLLM (default: gemini-2.5-pro-preview-03-25)."
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Use hardcoded debug parameters instead of CLI input."
    ),
):
    """
    Analyze a GitHub repository by sparse cloning, summarizing, and optionally generating an LLM-based summary.

    Examples:
        python gitgit.py analyze https://github.com/arangodb/python-arango --exts md,rst
        python gitgit.py analyze https://github.com/arangodb/python-arango --files README.md,docs/index.rst
        python gitgit.py analyze https://github.com/arangodb/python-arango --dirs docs/,src/ --summary --code-metadata --llm-model gpt-4o
    """
    main(
        repo_url=repo_url,
        extensions=extensions,
        files=files,
        dirs=dirs,
        debug=debug,
        summary=summary,
        code_metadata=code_metadata,
        llm_model=llm_model,
    )

def main(
    repo_url: str = "https://github.com/arangodb/python-arango",
    extensions: str = "md,rst",
    files: Optional[str] = None,
    dirs: Optional[str] = None,
    debug: bool = False,
    summary: bool = False,
    code_metadata: bool = False,
    llm_model: str = "gemini-2.5-pro-preview-03-25",
):
    """
    Main analysis workflow for a GitHub repository.

    - Sparse clones the repository and fetches only specified files, directories, or files with the specified extensions.
    - Concatenates and summarizes the content.
    - Optionally extracts code metadata (function names, parameters, docstrings).
    - Optionally generates an LLM-based summary using the specified model.

    Args:
        repo_url (str): GitHub repository URL.
        extensions (str): Comma-separated file extensions to include.
        files (str): Comma-separated specific file paths to include.
        dirs (str): Comma-separated directories to include.
        debug (bool): Use hardcoded debug parameters.
        summary (bool): Generate LLM summary if True.
        code_metadata (bool): Extract code metadata if True.
        llm_model (str): LLM model name for summarization.

    Usage Example:
        main("https://github.com/arangodb/python-arango", "md,rst,py", files="README.md", summary=True, code_metadata=True, llm_model="gpt-4o")
    """
    if debug:
        repo_url_ = "https://github.com/arangodb/python-arango"
        extensions_ = ["md", "rst"]
        files_ = None
        dirs_ = None
        logger.info(f"[DEBUG] Using hardcoded repo_url={repo_url_}, extensions={extensions_}")
    else:
        repo_url_ = repo_url
        extensions_ = [e.strip().lstrip('.') for e in extensions.split(',') if e.strip()]
        files_ = [f.strip() for f in files.split(',') if f.strip()] if files else None
        dirs_ = [d.strip() for d in dirs.split(',') if d.strip()] if dirs else None

    repo_name = repo_url_.rstrip('/').split('/')[-1]
    clone_dir = f"repos/{repo_name}_sparse"

    logger.info(f"Sparse cloning {repo_url_} for extensions: {extensions_}, files: {files_}, dirs: {dirs_} ...")
    sparse_clone(repo_url_, extensions_, clone_dir, files_, dirs_)

    found_files = debug_print_files(clone_dir, extensions_, files_, dirs_)
    logger.info(f"Files found after sparse checkout: {found_files}")

    logger.info(f"Running custom concatenation and summary...")
    summary_txt, tree, content = concat_and_summarize(clone_dir, extensions_, files_, dirs_, code_metadata)

    save_to_root(clone_dir, "SUMMARY.txt", summary_txt)
    save_to_root(clone_dir, "DIGEST.txt", content)
    save_to_root(clone_dir, "TREE.txt", tree)

    logger.info(f"\nSaved SUMMARY.txt, DIGEST.txt, and TREE.txt to {clone_dir}")

    if summary:
        logger.info("Running LLM summarization via LiteLLM...")
        llm_summarize(
            os.path.join(clone_dir, "DIGEST.txt"),
            os.path.join(clone_dir, "LLM_SUMMARY.txt"),
            model=llm_model,
        )

if __name__ == "__main__":
    app()