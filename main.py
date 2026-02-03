import os
import json
import logging
from typing import Dict, List

from dotenv import load_dotenv

from libs.git_kb_sync import GitSyncConfig, ensure_repo_at_branch
from libs.rag_knowledge_worker import RAGKnowledgeWorker

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_json(key: str, default: Dict) -> Dict:
    """Parse JSON from environment variable."""
    value = os.getenv(key)
    if value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {key}, using default")
    return default


def get_env_list(key: str, default: List[str]) -> List[str]:
    """Parse comma-separated list from environment variable."""
    value = os.getenv(key)
    if value:
        return [item.strip() for item in value.split(',') if item.strip()]
    return default


# Git configuration
GITLAB_TOKEN = os.getenv('GITLAB_TOKEN')
GITLAB_REPO_URL = os.getenv('GITLAB_REPO_URL', 'https://gitlab.com/your-username/your-repo.git')
GITLAB_BRANCH = os.getenv('GITLAB_BRANCH', 'main')
KB_FOLDER = os.getenv('KB_FOLDER', 'knowledge-base')

# RAG Configuration
FILE_PATTERNS = get_env_json('FILE_PATTERNS', {
    'cs': '**/*.cs',
    'md': '**/*.md',
    'csproj': '**/*.csproj',
    'sln': '**/*.sln'
})
MODEL = os.getenv('MODEL', 'llama3.2')
DB_NAME = os.getenv('DB_NAME', 'vector_db')
USE_OPENAI_EMBEDDINGS = get_env_bool('USE_OPENAI_EMBEDDINGS', False)
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '2000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '500'))
RETRIEVER_K = int(os.getenv('RETRIEVER_K', '25'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
FORCE_REFRESH_DB = get_env_bool('FORCE_REFRESH_DB', False)

# Deployment Configuration
SERVER_PORT = int(os.getenv('SERVER_PORT', '7860'))
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
OPEN_BROWSER = get_env_bool('OPEN_BROWSER', False)
SHARE_GRADIO = get_env_bool('SHARE_GRADIO', False)

# Excluded folders
EXCLUDED_FOLDERS = get_env_list('EXCLUDED_FOLDERS', [
    '.git', 'bin', 'obj', 'packages', 'node_modules', '.idea'
])

# Custom prompt (can be overridden via env)
DEFAULT_PROMPT = (
    "You are a helpful assistant specializing in your knowledge base.\n\n"
    "When answering:\n"
    "✓ Be accurate and cite sources from the knowledge base\n"
    "✓ Use code examples when helpful\n"
    "✓ Explain technical concepts clearly\n"
    "✗ Don't guess if the information isn't in the context\n"
    "✗ Don't provide outdated or speculative information\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
PROMPT = os.getenv('PROMPT_TEMPLATE', DEFAULT_PROMPT)


def sync_git_repo():
    """Sync the Git repository with the configured branch."""
    logger.info(f"Syncing repository: {GITLAB_REPO_URL} (branch: {GITLAB_BRANCH})")
    try:
        ensure_repo_at_branch(
            target_folder=KB_FOLDER,
            config=GitSyncConfig(
                repo_url=GITLAB_REPO_URL,
                branch=GITLAB_BRANCH,
                token=GITLAB_TOKEN,
            ),
        )
        logger.info("Repository synced successfully")
    except RuntimeError as e:
        logger.error(f"Failed to sync repository: {e}")
        raise SystemExit(1)


def initialize_rag() -> RAGKnowledgeWorker:
    """Initialize the RAG Knowledge Worker."""
    logger.info("Initializing RAG Knowledge Worker")

    rag = RAGKnowledgeWorker(
        kb_folder=KB_FOLDER,
        file_patterns=FILE_PATTERNS,
        model_name=MODEL,
        db_name=DB_NAME,
        use_openai_embeddings=USE_OPENAI_EMBEDDINGS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        retriever_k=RETRIEVER_K,
        temperature=TEMPERATURE,
        prompt=PROMPT,
        excluded_folders=EXCLUDED_FOLDERS
    )

    logger.info(f"Initializing RAG system (force_refresh={FORCE_REFRESH_DB})")
    rag.initialize(force_refresh_db=FORCE_REFRESH_DB)
    logger.info("RAG system initialized successfully")

    return rag


def main():
    """Main entry point for the application."""
    logger.info("Starting Code Base Support Agent")
    logger.info(f"Configuration: Model={MODEL}, DB={DB_NAME}, Port={SERVER_PORT}")

    # Sync repository
    sync_git_repo()

    # Initialize RAG
    rag = initialize_rag()

    # Launch Gradio interface
    logger.info(f"Launching Gradio interface on {SERVER_HOST}:{SERVER_PORT}")
    rag.launch_gradio(
        inbrowser=OPEN_BROWSER,
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        share=SHARE_GRADIO
    )


if __name__ == "__main__":
    main()
