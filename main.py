from libs.rag_knowledge_worker import RAGKnowledgeWorker, create_rag_worker
import subprocess
import os
from dotenv import load_dotenv


load_dotenv(override=True)

# Git configuration
GITLAB_TOKEN = os.getenv('GITLAB_TOKEN')
GITLAB_REPO_URL = os.getenv('GITLAB_REPO_URL', 'https://gitlab.com/your-username/your-repo.git')
GITLAB_BRANCH = os.getenv('GITLAB_BRANCH', 'main')  # Default to 'main' branch
KB_FOLDER = 'knowledge-base'

# Construct authenticated URL (token method)
if GITLAB_TOKEN:
    auth_url = GITLAB_REPO_URL.replace('https://', f'https://oauth2:{GITLAB_TOKEN}@')
else:
    auth_url = GITLAB_REPO_URL

# Git checkout/clone knowledge base
if os.path.exists(KB_FOLDER):
    try:
        print(f"Updating {KB_FOLDER} from git (branch: {GITLAB_BRANCH})...")
        subprocess.run(['git', '-C', KB_FOLDER, 'fetch', 'origin'], check=True)
        subprocess.run(['git', '-C', KB_FOLDER, 'checkout', GITLAB_BRANCH], check=True)
        subprocess.run(['git', '-C', KB_FOLDER, 'pull', 'origin', GITLAB_BRANCH], check=True)
        print(f"Successfully updated {KB_FOLDER} on branch {GITLAB_BRANCH}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not update git repository: {e}")
else:
    try:
        print(f"Cloning repository into {KB_FOLDER} (branch: {GITLAB_BRANCH})...")
        subprocess.run(['git', 'clone', '-b', GITLAB_BRANCH, auth_url, KB_FOLDER], check=True)
        print(f"Successfully cloned {KB_FOLDER} on branch {GITLAB_BRANCH}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Could not clone git repository: {e}")
        exit(1)


# Configuration
FILE_PATTERNS = {
    'cs': '**/*.cs',
    'md': '**/*.md',
    'csproj': '**/*.csproj',
    'sln': '**/*.sln'
}
MODEL = "gpt-4o-mini" #"gpt-4o-mini"   or "llama3.2" for Ollama
DB_NAME = "vector_db"
PROMPT = ("You are a helpful coding assistant specializing in Vumatel's NMSG (Network Management System Gateway) API.\n\n"
          "When answering:\n"
          "✓ Be accurate and cite sources from the codebase\n"
          "✓ Use code examples when helpful\n"
          "✓ Explain technical concepts clearly\n"
          "✗ Don't guess if the information isn't in the context\n"
          "✗ Don't provide outdated or speculative information\n\n"
          "Context: {context}\n\n"
          "Question: {question}\n\n"
          "Answer:")


rag = RAGKnowledgeWorker(
    kb_folder=KB_FOLDER,
    file_patterns=FILE_PATTERNS,
    model_name=MODEL,
    db_name=DB_NAME,
    use_openai_embeddings=True, # set too False to use Huggingface embeddings
    chunk_size=2000,
    chunk_overlap=500,
    retriever_k=25,
    temperature=0.7,
    prompt=PROMPT,
    excluded_folders=['.git', 'bin', 'obj', 'packages', 'node_modules', '.idea', 'NmsGateway.Tests']
)

# Initialize the system
rag.initialize(force_refresh_db=True)

# Option C: Launch Gradio interface
rag.launch_gradio(inbrowser=True)

