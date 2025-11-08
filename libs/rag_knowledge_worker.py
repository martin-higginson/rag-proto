"""
RAG Knowledge Worker Module

A configurable, reusable module for building RAG (Retrieval Augmented Generation) 
systems with support for multiple LLM providers and embedding options.

This module provides a high-level interface for creating knowledge-based question-answering
systems using vector databases and large language models. It supports both cloud-based
(OpenAI) and local (Ollama) LLM providers, with flexible embedding options.

Key Features:
    - Multi-provider LLM support (OpenAI, Ollama)
    - Flexible embedding options (OpenAI, HuggingFace)
    - Vector database integration with Chroma
    - Built-in Gradio interface for quick prototyping
    - Customizable document processing pipeline
    - Recursive directory loading with multiple file type support

Example:
    Basic usage with OpenAI:
    
    >> from libs.rag_knowledge_worker import RAGKnowledgeWorker
    >> 
    >> rag = RAGKnowledgeWorker(
    ...     kb_folder="path/to/knowledge/base",
    ...     file_patterns={'py': '**/*.py', 'md': '**/*.md'},
    ...     model_name="gpt-4o-mini",
    ...     db_name="my_vector_db"
    ... )
    >> rag.initialize()
    >> answer = rag.query("What is this codebase about?")
    >> print(answer)
    
    Using local Ollama model:
    
    >> rag = RAGKnowledgeWorker(
    ...     kb_folder="docs/",
    ...     file_patterns={'txt': '**/*.txt'},
    ...     model_name="llama3.2",
    ...     use_openai_embeddings=False
    ... )
    >> rag.initialize()
    >> rag.launch_gradio()

Dependencies:
    - langchain-community: Document loading and retrieval
    - langchain-openai: OpenAI LLM and embeddings
    - langchain-chroma: Vector database
    - gradio: Web interface
    - python-dotenv: Environment variable management

Author: Ed Donner (adapted for course materials)
"""

import os
from typing import Dict, Optional, List, Callable
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Model name patterns for automatic provider detection
OLLAMA_PATTERNS = [
    'llama3.2', 'llama3.1', 'llama3', 'llama2', 'llama',
    'mistral', 'mixtral', 'phi3', 'phi', 'gemma',
    'codellama', 'neural-chat', 'orca', 'vicuna',
    'wizard', 'nous-hermes', 'openchat', 'starling',
    'solar', 'dolphin', 'samantha', 'tinyllama'
]

OPENAI_PATTERNS = [
    'gpt-4o', 'gpt-4', 'gpt-3.5', 'gpt-35',
    'text-davinci', 'text-curie', 'text-babbage', 'text-ada'
]


class RAGKnowledgeWorker:
    """
    A configurable RAG (Retrieval Augmented Generation) system that works with
    multiple LLM providers and embedding options.
    
    This class orchestrates the entire RAG pipeline including document loading,
    text splitting, embedding generation, vector storage, and question answering.
    It automatically detects the appropriate LLM provider based on the model name.
    
    Attributes:
        kb_folder (str): Path to the knowledge base directory
        file_patterns (Dict[str, str]): Mapping of file types to glob patterns
        model_name (str): Name of the LLM model
        db_name (str): Path for the vector database
        use_openai_embeddings (bool): Whether to use OpenAI embeddings
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Overlap between consecutive chunks
        retriever_k (int): Number of chunks to retrieve for context
        temperature (float): LLM generation temperature
        ollama_base_url (str): Base URL for Ollama API
        
    Example:
       >> rag = RAGKnowledgeWorker(
        ...     kb_folder="./docs",
        ...     file_patterns={'md': '**/*.md'},
        ...     model_name="gpt-4o-mini"
        ... )
       >> rag.initialize()
       >> answer = rag.query("Explain the main concept")
    """

    def __init__(
            self,
            kb_folder: str,
            file_patterns: Dict[str, str],
            model_name: str = "gpt-4o-mini",
            db_name: str = "vector_db",
            use_openai_embeddings: bool = True,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            retriever_k: int = 25,
            temperature: float = 0.7,
            text_loader_kwargs: Optional[Dict] = None,
            ollama_base_url: str = 'http://localhost:11434/v1',
            prompt: Optional[str] = None,
            excluded_folders: Optional[List[str]] = None
    ):
        """
        Initialize the RAG Knowledge Worker.

        Args:
            kb_folder: Path to the knowledge base folder containing documents
            file_patterns: Dictionary mapping file types to glob patterns
                          Example: {'cs': '**/*.cs', 'md': '**/*.md'}
            model_name: Name of the LLM model to use (OpenAI or Ollama)
            db_name: Name/path for the vector database storage
            use_openai_embeddings: If True, use OpenAI embeddings; if False, use HuggingFace
            chunk_size: Maximum size of text chunks (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            retriever_k: Number of relevant chunks to retrieve for each query
            temperature: LLM generation temperature (0.0 = deterministic, 1.0 = creative)
            text_loader_kwargs: Additional arguments for TextLoader
                               Example: {'encoding': 'utf-8', 'autodetect_encoding': True}
            ollama_base_url: Base URL for local Ollama API
            prompt: Optional custom prompt template string. If None, uses default RAG prompt.
                    Should include {context} and {question} placeholders.
                    Example: "Use this context: {context}\n\nQuestion: {question}\n\nAnswer:"
            excluded_folders: Optional list of folder names to exclude from the knowledge base.
                            Folders are matched by name (not full path).
                            Example: ['.git', 'node_modules', 'build', 'dist']

        Note:
            - File patterns support recursive glob patterns (e.g., '**/*.py')
            - For Windows encoding issues, try text_loader_kwargs={'autodetect_encoding': True}
            - Lower temperature values produce more focused, deterministic responses
            - Excluded folders apply to any subdirectory with that name under kb_folder
        """
        self.kb_folder = kb_folder
        self.file_patterns = file_patterns
        self.model_name = model_name
        self.db_name = db_name
        self.use_openai_embeddings = use_openai_embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k
        self.temperature = temperature
        self.ollama_base_url = ollama_base_url
        self.custom_prompt_str = prompt
        self.excluded_folders = excluded_folders or []

        self.text_loader_kwargs = text_loader_kwargs or {'encoding': 'utf-8'}

        # Components initialized later
        self.chunks: Optional[List[Document]] = None
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.rag_chain = None
        self.prompt_template: Optional[ChatPromptTemplate] = None

        load_dotenv(override=True)

    def create_llm(self) -> ChatOpenAI:
        """
        Create an LLM instance with automatic provider detection.
        
        Automatically detects whether to use OpenAI or Ollama based on the model name.
        OpenAI models are identified by patterns like 'gpt-4', 'gpt-3.5', etc.
        Ollama models are identified by patterns like 'llama', 'mistral', etc.
        
        Returns:
            ChatOpenAI: Configured LLM instance for the appropriate provider
            
        Note:
            - For OpenAI: Requires OPENAI_API_KEY environment variable
            - For Ollama: Requires local Ollama server running at ollama_base_url
            - Ollama doesn't require an API key
            
        Example:
           >> rag = RAGKnowledgeWorker(kb_folder="docs", file_patterns={}, model_name="llama3.2")
           >> llm = rag.create_llm()  # Returns Ollama-configured ChatOpenAI
        """
        model_lower = self.model_name.lower()

        is_ollama = any(pattern in model_lower for pattern in OLLAMA_PATTERNS)
        is_openai = any(pattern in model_lower for pattern in OPENAI_PATTERNS)

        if is_ollama and not is_openai:
            print(f"🦙 Using Ollama with model: {self.model_name}")
            return ChatOpenAI(
                temperature=self.temperature,
                model=self.model_name,
                base_url=self.ollama_base_url,
                api_key=None
            )
        else:
            print(f"🤖 Using OpenAI with model: {self.model_name}")
            return ChatOpenAI(
                temperature=self.temperature,
                model=self.model_name
            )

    @staticmethod
    def add_metadata(doc: Document, doc_type: str) -> Document:
        """
        Add document type metadata to a document.
        
        Args:
            doc: Document to add metadata to
            doc_type: Type/category of the document
            
        Returns:
            Document: The document with added metadata
        """
        doc.metadata["doc_type"] = doc_type
        return doc

    def load_knowledge_base(self) -> tuple:
        """
        Load documents from the knowledge base and create embeddings.
    
        Recursively loads all files matching the specified patterns from the
        knowledge base folder, splits them into chunks, and prepares embedding
        function for vectorization. Excludes any folders specified in excluded_folders.
    
        Returns:
            tuple: (chunks, embeddings) where:
                - chunks: List of Document objects split into manageable pieces
                - embeddings: Embedding function (OpenAI or HuggingFace)
            
        Raises:
            Exception: Prints warning if specific file types fail to load
        
        Note:
            - Files are loaded recursively from all subdirectories
            - Each document is tagged with its file type in metadata
            - Failed file loads are non-fatal and print warnings
            - Folders in excluded_folders list are skipped during loading
        
        Example:
            >> chunks, embeddings = rag.load_knowledge_base()
            >> print(f"Loaded {len(chunks)} chunks")
        """
        documents = []

        print(f"Loading files from {self.kb_folder} and all subdirectories...")
        if self.excluded_folders:
            print(f"Excluding folders: {', '.join(self.excluded_folders)}")

        for file_type, pattern in self.file_patterns.items():
            try:
                loader = DirectoryLoader(
                    self.kb_folder,
                    glob=pattern,
                    loader_cls=TextLoader,
                    loader_kwargs=self.text_loader_kwargs,
                    recursive=True,
                    exclude=self.excluded_folders
                )
                type_docs = loader.load()
                documents.extend([self.add_metadata(doc, file_type) for doc in type_docs])
                print(f"Loaded {len(type_docs)} {file_type} files")
            except Exception as e:
                print(f"Warning: Error loading {file_type} files: {e}")

        print(f"Total documents loaded: {len(documents)}")

        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)

        if self.use_openai_embeddings:
            print("Using OpenAI embeddings")
            embeddings = OpenAIEmbeddings()
        else:
            print("Using HuggingFace embeddings")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        return chunks, embeddings

    def create_vector_store(self, force_refresh: bool = False) -> Chroma:
        """
        Create or load the vector store database.
        
        Creates a Chroma vector database from the document chunks and embeddings.
        Can optionally delete and recreate an existing database.
        
        Args:
            force_refresh: If True, delete existing collection and create new one
            
        Returns:
            Chroma: Vector store instance containing embedded documents
            
        Note:
            - Database is persisted to disk at self.db_name location
            - Force refresh is useful when knowledge base has changed significantly
            - Creation time depends on number of chunks and embedding provider
            
        Example:
            >> vectorstore = rag.create_vector_store(force_refresh=True)
            >> print(f"Vector store has {vectorstore._collection.count()} documents")
        """
        if force_refresh and os.path.exists(self.db_name):
            print(f"Deleting existing vector store: {self.db_name}")
            Chroma(
                persist_directory=self.db_name,
                embedding_function=self.embeddings
            ).delete_collection()

        store = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=self.db_name
        )
        print(f"Vectorstore created with {store._collection.count()} documents")

        return store

    def format_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a single context string.
        
        Args:
            docs: List of Document objects to format
            
        Returns:
            str: Documents concatenated with double newlines as separators
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def create_rag_prompt(self) -> ChatPromptTemplate:
        """
        Create the default RAG prompt template.
    
        Returns:
            ChatPromptTemplate: Template for RAG question answering with context
        
        Note:
            If a custom prompt string was provided during initialization, it will be used.
            Otherwise, the default prompt instructs the model to answer questions based only
            on the provided context. You can customize this by passing a different
            template to create_rag_chain().
        
        Example:
            Custom prompt during initialization:
           >> rag = RAGKnowledgeWorker(
            ...     kb_folder="docs",
            ...     file_patterns={'md': '**/*.md'},
            ...     prompt="Context: {context}\n\nQ: {question}\n\nA:"
            ... )
            
            Custom prompt at chain creation:
           >> custom_prompt = ChatPromptTemplate.from_template('''
            ... Answer in detail using this context: {context}
            ... Question: {question}
            ... Detailed Answer:
            ... ''')
           >> rag.create_rag_chain(custom_prompt=custom_prompt)
        """
        if self.custom_prompt_str:
            return ChatPromptTemplate.from_template(self.custom_prompt_str)

        return ChatPromptTemplate.from_template("""Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:""")

    def create_rag_chain(self, custom_prompt: Optional[ChatPromptTemplate] = None):
        """
        Create the RAG chain for question answering.
        
        Constructs a LangChain pipeline that retrieves relevant documents,
        formats them as context, applies a prompt template, queries the LLM,
        and parses the output.
        
        Args:
            custom_prompt: Optional custom prompt template. If None, uses default.
        
        Returns:
            Runnable: A chain that can be invoked with questions
            
        Note:
            The chain structure is:
            1. Retrieve relevant documents from vector store
            2. Format documents as context
            3. Apply prompt template with context and question
            4. Query LLM
            5. Parse string output
            
        Example:
           >> rag.initialize()
           >> answer = rag.rag_chain.invoke("What is the main topic?")
        """
        prompt = custom_prompt or self.prompt_template or self.create_rag_prompt()

        rag_chain = (
                {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return rag_chain

    def initialize(self, force_refresh_db: bool = False):
        """
        Initialize the complete RAG system.
        
        Performs the full initialization sequence: loads knowledge base, creates
        vector store, sets up LLM, and builds the RAG chain. This must be called
        before querying the system.
        
        Args:
            force_refresh_db: If True, delete and recreate the vector database
            
        Raises:
            Exception: May raise exceptions from underlying components if
                      initialization fails (e.g., missing API keys, file errors)
            
        Note:
            - This method can take several minutes for large knowledge bases
            - force_refresh_db=True is useful when documents have changed
            - Progress is printed to console during initialization
            
        Example:
           >> rag = RAGKnowledgeWorker(
            ...     kb_folder="./docs",
            ...     file_patterns={'md': '**/*.md'},
            ...     model_name="gpt-4o-mini"
            ... )
           >> rag.initialize()  # Takes time on first run
           >> rag.initialize(force_refresh_db=False)  # Fast on subsequent runs
        """
        print("Initializing RAG Knowledge Worker...")

        self.chunks, self.embeddings = self.load_knowledge_base()
        self.vectorstore = self.create_vector_store(force_refresh=force_refresh_db)
        self.llm = self.create_llm()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.retriever_k})
        self.prompt_template = self.create_rag_prompt()
        self.rag_chain = self.create_rag_chain()

        print("✅ RAG Knowledge Worker initialized successfully!")

    async def aquery(self, question: str) -> str:
        """
        Async query the RAG system with a question.

        Args:
            question: The question to ask the RAG system

        Returns:
            str: The generated answer based on retrieved context

        Raises:
            RuntimeError: If the system hasn't been initialized

        Example:
           >> answer = await rag.aquery("What is the purpose of this codebase?")
           >> print(answer)
        """
        if self.rag_chain is None:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        return await self.rag_chain.ainvoke(question)

    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask the RAG system
            
        Returns:
            str: The generated answer based on retrieved context
            
        Raises:
            RuntimeError: If the system hasn't been initialized
            
        Example:
           >> answer = rag.query("What is the purpose of this codebase?")
           >> print(answer)
        """
        if self.rag_chain is None:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        return self.rag_chain.invoke(question)

    async def achat(self, question: str, history: List = None) -> str:
        """
        Async chat interface method for Gradio compatibility.

        Provides an async chat interface that wraps the aquery method.
        The history parameter is accepted but not used in the current implementation.

        Args:
            question: The question to ask
            history: Chat history (for Gradio interface, currently unused)

        Returns:
            str: The generated answer

        Note:
            Future versions may incorporate chat history for context-aware conversations.
        """
        return await self.aquery(question)

    def chat(self, question: str, history: List = None) -> str:
        """
        Chat interface method for Gradio compatibility.
        
        Provides a simple chat interface that wraps the query method.
        The history parameter is accepted but not used in the current implementation.
        
        Args:
            question: The question to ask
            history: Chat history (for Gradio interface, currently unused)
            
        Returns:
            str: The generated answer
            
        Note:
            Future versions may incorporate chat history for context-aware conversations.
        """
        return self.query(question)

    def launch_gradio(self, inbrowser: bool = True, share: bool = False, **kwargs):
        """
        Launch an interactive Gradio chat interface.

        Creates and launches a web-based chat interface for interacting with
        the RAG system. Useful for quick prototyping and demos.

        Args:
            inbrowser: Whether to automatically open in browser
            share: Whether to create a public shareable link
            **kwargs: Additional arguments passed to gr.ChatInterface.launch()

        Returns:
            Gradio interface instance

        Raises:
            RuntimeError: If the system hasn't been initialized

        Note:
            - The interface runs on localhost by default
            - Setting share=True creates a temporary public URL
            - The server runs until manually stopped (Ctrl+C)

        Example:
           >> rag.initialize()
           >> rag.launch_gradio(inbrowser=True, share=False)
            # Opens browser with chat interface
        """
        if self.rag_chain is None:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        interface = gr.ChatInterface(
            self.achat,  # Use async version
            type="messages"
        )

        return interface.launch(inbrowser=inbrowser, share=share, **kwargs)

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents without generating an answer.
        
        Useful for debugging, understanding what context is being used, or
        building custom RAG pipelines.
        
        Args:
            query: The query text to search for
            k: Number of documents to retrieve. If None, uses self.retriever_k
            
        Returns:
            List[Document]: Relevant documents with metadata
            
        Raises:
            RuntimeError: If the system hasn't been initialized
            
        Example:
           >> docs = rag.get_relevant_documents("machine learning", k=5)
           >> for doc in docs:
            ...     print(f"Source: {doc.metadata['source']}")
            ...     print(f"Content: {doc.page_content[:100]}...")
        """
        if self.retriever is None:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        if k is not None:
            temp_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            return temp_retriever.get_relevant_documents(query)

        return self.retriever.get_relevant_documents(query)


def create_rag_worker(
        kb_folder: str,
        file_patterns: Dict[str, str],
        model_name: str = "gpt-4o-mini",
        prompt: Optional[str] = None,
        **kwargs
) -> RAGKnowledgeWorker:
    """
    Convenience function to create and initialize a RAG Knowledge Worker.

    This is a shortcut that combines instantiation and initialization in one call.

    Args:
        kb_folder: Path to the knowledge base folder
        file_patterns: Dictionary of file type patterns
        model_name: Name of the LLM model to use
        prompt: Optional custom prompt template string. If None, uses default RAG prompt.
                Should include {context} and {question} placeholders.
                Example: "Use this context: {context}\n\nQuestion: {question}\n\nAnswer:"
        **kwargs: Additional configuration parameters passed to RAGKnowledgeWorker
                 Including excluded_folders: List[str] to exclude specific folders
    
    Returns:
        RAGKnowledgeWorker: Fully initialized instance ready to use
    
    Example:
        Quick setup:
        >> rag = create_rag_worker(
        ...     kb_folder="./documentation",
        ...     file_patterns={'md': '**/*.md', 'txt': '**/*.txt'},
        ...     model_name="gpt-4o-mini",
        ...     retriever_k=20
        ... )
        >> answer = rag.query("How do I configure the system?")
    
        With custom prompt:
        >> rag = create_rag_worker(
        ...     kb_folder="./code",
        ...     file_patterns={'py': '**/*.py'},
        ...     model_name="gpt-4o-mini",
        ...     prompt="You are a code expert. Context: {context}\n\nQuestion: {question}\n\nDetailed Answer:"
        ... )
        >> answer = rag.query("Explain the main function")
    
        With excluded folders:
        >> rag = create_rag_worker(
        ...     kb_folder="./code",
        ...     file_patterns={'py': '**/*.py'},
        ...     model_name="gpt-4o-mini",
        ...     excluded_folders=['.git', 'node_modules', '__pycache__', 'venv']
        ... )
        >> answer = rag.query("Explain the main function")

        With local Ollama:
        >> rag = create_rag_worker(
        ...     kb_folder="./code",
        ...     file_patterns={'py': '**/*.py'},
        ...     model_name="llama3.2",
        ...     use_openai_embeddings=False
        ... )
        >> rag.launch_gradio()
    """
    worker = RAGKnowledgeWorker(
        kb_folder=kb_folder,
        file_patterns=file_patterns,
        model_name=model_name,
        prompt=prompt,
        **kwargs
    )
    worker.initialize()
    return worker
