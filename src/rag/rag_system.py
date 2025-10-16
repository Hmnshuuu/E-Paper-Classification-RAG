"""
RAG (Retrieval-Augmented Generation) System
Enables retrieval of relevant historical articles and generation of summaries
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document


class RAGSystem:
    """
    Retrieval-Augmented Generation system for e-paper content.

    Addresses requirements:
    1. Retrieve relevant historical or contextual articles from vector database
    2. Generate summaries and metadata
    3. Support for multilingual content
    4. Large-scale deployment across multiple newspaper editions
    """

    def __init__(self, persist_directory='../data/chroma_db', embedding_model=None):
        """
        Initialize RAG system with vector store.

        Args:
            persist_directory: Path to store the vector database
            embedding_model: Name of sentence-transformer model to use
                           (default: all-MiniLM-L6-v2 - good balance of speed and quality)
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embeddings
        if embedding_model is None:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

        print(f"Initializing embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}  # Can be changed to 'cuda' if GPU available
        )

        # Text splitter for chunking large articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # ~500 characters per chunk
            chunk_overlap=50,     # 50 char overlap to maintain context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.vectorstore = None

    def create_vectorstore(self, documents, show_progress=True):
        """
        Create vector store from processed newspaper regions.

        Args:
            documents: List of region dictionaries with 'text', 'classification', etc.
            show_progress: Whether to print progress updates

        Returns:
            Number of document chunks created
        """
        if not documents:
            print("Warning: No documents provided to create vectorstore")
            return 0

        # Filter out empty text documents
        valid_docs = [doc for doc in documents if doc.get('text', '').strip()]

        if not valid_docs:
            print("Warning: No valid documents with text found")
            return 0

        if show_progress:
            print(f"Creating vectorstore from {len(valid_docs)} documents...")

        # Create Document objects for LangChain
        langchain_docs = []

        for i, doc in enumerate(valid_docs):
            text = doc['text']

            # Create metadata
            metadata = {
                'classification': doc.get('classification', 'unknown'),
                'page': doc.get('page', 0),
                'type': doc.get('type', 'unknown'),
                'confidence': doc.get('classification_confidence', 0.0),
                'language': doc.get('detected_language', 'unknown')
            }

            # Split long documents into chunks
            chunks = self.text_splitter.split_text(text)

            for j, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = j
                chunk_metadata['total_chunks'] = len(chunks)

                langchain_docs.append(
                    Document(page_content=chunk, metadata=chunk_metadata)
                )

        if show_progress:
            print(f"Created {len(langchain_docs)} text chunks")
            print("Building vector index...")

        # Create Chroma vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Persist to disk
        self.vectorstore.persist()

        if show_progress:
            print(f"✓ Vector store created and persisted to {self.persist_directory}")

        return len(langchain_docs)

    def load_vectorstore(self):
        """
        Load existing vector store from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"✓ Vector store loaded from {self.persist_directory}")
            return True
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return False

    def retrieve(self, query, k=5, filter_news=True, filter_language=None):
        """
        Retrieve relevant documents based on semantic similarity.

        Args:
            query: Search query string
            k: Number of results to return
            filter_news: If True, only return news articles (not ads)
            filter_language: Optional language filter ('en', 'hi', etc.)

        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load first.")

        # Build filter dictionary
        filter_dict = {}
        if filter_news:
            filter_dict['classification'] = 'news article'
        if filter_language:
            filter_dict['language'] = filter_language

        # Perform similarity search
        results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter_dict if filter_dict else None
        )

        return results

    def retrieve_with_scores(self, query, k=5, filter_news=True):
        """
        Retrieve documents with similarity scores.

        Args:
            query: Search query
            k: Number of results
            filter_news: Filter for news articles only

        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        filter_dict = {'classification': 'news article'} if filter_news else None

        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict
        )

        return results

    def generate_summary(self, documents, max_sentences=3):
        """
        Generate extractive summary from retrieved documents.

        This is a simple extractive summarization approach.
        For production, could be enhanced with:
        - Abstractive summarization using T5/BART
        - Multi-document summarization
        - Language-specific summarization

        Args:
            documents: List of Document objects
            max_sentences: Maximum sentences in summary

        Returns:
            Summary string
        """
        if not documents:
            return "No relevant documents found."

        # Combine all document content
        combined_text = "\n\n".join([doc.page_content for doc in documents])

        # Simple extractive summary: take first N sentences
        sentences = []
        for sep in ['. ', '। ', '॥ ']:  # English, Hindi punctuation
            if sep in combined_text:
                sentences = combined_text.split(sep)
                break

        if not sentences:
            sentences = [combined_text]

        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Take first max_sentences
        summary_sentences = sentences[:max_sentences]
        summary = '. '.join(summary_sentences)

        if summary and not summary.endswith('.'):
            summary += '.'

        return summary

    def generate_metadata(self, documents):
        """
        Generate metadata from retrieved documents.

        Returns:
            Dictionary with aggregated metadata:
            - Topics/categories
            - Date coverage
            - Languages
            - Average confidence scores
        """
        if not documents:
            return {}

        metadata = {
            'document_count': len(documents),
            'pages': set(),
            'types': set(),
            'languages': set(),
            'classifications': {},
            'avg_confidence': 0.0
        }

        confidences = []

        for doc in documents:
            meta = doc.metadata

            metadata['pages'].add(meta.get('page', 'unknown'))
            metadata['types'].add(meta.get('type', 'unknown'))
            metadata['languages'].add(meta.get('language', 'unknown'))

            classification = meta.get('classification', 'unknown')
            metadata['classifications'][classification] = \
                metadata['classifications'].get(classification, 0) + 1

            conf = meta.get('confidence', 0)
            if conf > 0:
                confidences.append(conf)

        if confidences:
            metadata['avg_confidence'] = sum(confidences) / len(confidences)

        # Convert sets to lists for JSON serialization
        metadata['pages'] = sorted(list(metadata['pages']))
        metadata['types'] = list(metadata['types'])
        metadata['languages'] = list(metadata['languages'])

        return metadata

    def query(self, query, k=5, include_metadata=True, include_summary=True):
        """
        High-level query interface that returns everything.

        Args:
            query: Search query
            k: Number of results
            include_metadata: Whether to generate metadata
            include_summary: Whether to generate summary

        Returns:
            Dictionary with:
            - documents: Retrieved documents
            - summary: Extractive summary
            - metadata: Aggregated metadata
            - query: Original query
        """
        results = self.retrieve(query, k=k)

        response = {
            'query': query,
            'documents': results,
            'document_count': len(results)
        }

        if include_summary:
            response['summary'] = self.generate_summary(results)

        if include_metadata:
            response['metadata'] = self.generate_metadata(results)

        return response

    def get_statistics(self):
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with vectorstore statistics
        """
        if self.vectorstore is None:
            return {"status": "not initialized"}

        try:
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()

            return {
                "status": "active",
                "total_chunks": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embeddings.model_name
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
