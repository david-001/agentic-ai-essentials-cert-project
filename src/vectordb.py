import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", config.CHROMA_COLLECTION_NAME
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", config.EMBEDDING_MODEL
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

        Chunk size is set to 512 characters with 50-character overlap by default,
        which balances context preservation with retrieval precision for the
        TaskFlow Pro markdown documents. RecursiveCharacterTextSplitter respects
        sentence and paragraph boundaries, avoiding mid-sentence splits.

        Args:
            text: Input text to chunk
            chunk_size: Maximum number of characters per chunk (default: 512)
            chunk_overlap: Number of overlapping characters between chunks (default: 50)

        Returns:
            List of text chunks
        """
        # Use LangChain's RecursiveCharacterTextSplitter
        #   - From langchain_text_splitters import RecursiveCharacterTextSplitter
        #   - Automatically handles sentence boundaries and preserves context better

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # Implement document ingestion logic
        #   - Loop through each document in the documents list
        #   - Extract 'content' and 'metadata' from each document dict
        #   - Use self.chunk_text() to split each document into chunks
        #   - Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        #   - Use self.embedding_model.encode() to create embeddings for all chunks
        #   - Store the embeddings, documents, metadata, and IDs in your vector database
        #   - Print progress messages to inform the user

        print(f"Processing {len(documents)} documents...")

        # Handle empty document list
        if not documents:
            print("No documents to process.")
            return
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        # Process each document
        for doc_idx, document in enumerate(documents):
            # Extract content and metadata
            content = document.get('content', '')
            metadata = document.get('metadata', {})
            
            # Chunk the document
            chunks = self.chunk_text(content)
            print(f"Document {doc_idx + 1}: Split into {len(chunks)} chunks")
            
            # Create unique IDs and metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                chunk_metadata = {
                    **metadata,
                    'doc_index': doc_idx,
                    'chunk_index': chunk_idx
                }
                
                all_chunks.append(chunk)
                all_metadatas.append(chunk_metadata)
                all_ids.append(chunk_id)
        
        if not all_chunks:
            print("No chunks to add!")
            return
        
        # Create embeddings for all chunks
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Add to ChromaDB collection
        print("Adding to vector database...")
        self.collection.add(
            ids=all_ids,
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadatas
        )
        
        print(f"Successfully added {len(all_chunks)} chunks to vector database")

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # Implement similarity search logic
        #   - Use self.embedding_model.encode([query]) to create query embedding
        #   - Convert the embedding to appropriate format for your vector database
        #   - Use your vector database's search/query method with the query embedding and n_results
        #   - Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        #   - Handle the case where results might be empty

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        # ChromaDB returns results in a specific format
        # Extract the actual results (they're in lists, take the first element)
        return {
            "documents": results['documents'][0] if results['documents'] else [],
            "metadatas": results['metadatas'][0] if results['metadatas'] else [],
            "distances": results['distances'][0] if results['distances'] else [],
            "ids": results['ids'][0] if results['ids'] else [],
        }
