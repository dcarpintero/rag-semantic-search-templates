# Semantic Search and Retrieval-Augmented Generation in Practice

Retrieval-Augmented Generation (RAG) is a powerful approach that enhances Large Language Models (LLMs) by combining their generative capabilities with the ability to access and incorporate external knowledge during the response generation process.

Semantic Search goes beyond traditional keyword matching by understanding the meaning and intent behind search queries - it uses dense vector representations (embeddings) to capture the semantic meaning of both the query and the documents in the knowledge base, allowing it to find relevant information even when the exact keywords do not match. When integrated with RAG, Semantic Search serves as an advanced retrieval mechanism that helps the LLM find the most contextually relevant information from a custom knowledge base.

## ETL

The first step when implementing a RAG system with Semantic Search is to prepare a knowledge base. This involves collecting and preprocessing documents - these could be web pages, PDFs, internal docs, or any text-based information.

## Vectorization into High-Dimensional Embeddings

### Understanding Embeddings

Embeddings represent data (such as text) as numerical vectors in high-dimensional space, typically using 512, 768, or 1024 dimensions. These dense vectors capture the semantic meaning and relationships between different pieces of data, enabling more nuanced and context-aware matching and retrieval. Think of them as sophisticated mathematical fingerprints that preserve the essence and meaning of the original text.

### Document Chunking

In practice, documents are first divided into smaller, meaningful chunks before converted them into vectors. This chunking process represents a critical design decision in RAG-based applications, as it requires careful balancing of context and specificity. If chunks are too large, the embeddings might become too general and lose important details; if they're too small, they might lose the contextual relationships that give text its deeper meaning.

### Chunking Strategies and Metadata Enhancement

The optimal chunking strategy depends heavily on the nature of your texts and the types of queries your system needs to handle. Common approaches include dividing text at natural boundaries like sentences or paragraphs, often with overlapping segments to preserve context across chunk boundaries. Many implementations also enhance each chunk by incorporating document metadata, such as section titles or document headings, which provides additional context and improves retrieval accuracy. This metadata enrichment helps maintain the hierarchical structure and topical relationships present in the original document.

### Frameworks for Text Embeddings

Text embedding frameworks provide the essential tools for converting text into high-dimensional vectors.

#### Sentence Transformers

[SentenceTransformers](https://huggingface.co/sentence-transformers) is a Python framework for state-of-the-art sentence, text and image embeddings.

The following example demonstrates how to generate embeddings from academic papers using SentenceTransformers. We'll use the `jinaai/jina-embeddings-v2-base-en` model to vectorize `title, abstract` pairs from an arXiv dataset with `batch_size = 64`. This allows for parallel computation on hardware accelerators like GPUs (albeit at the cost of requiring more memory): 

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

from datasets import load_dataset
ds = load_dataset("dcarpintero/arxiv.cs.CL.25k", split="train")

corpus = [title + ':' + abstract for title, abstract in zip(ds['title'], ds['abstract'])]
f32_embeddings = model.encode(corpus,
                              batch_size=64,
                              show_progress_bar=True)
```

### LlamaIndex

[LlamaIndex](https://docs.llamaindex.ai/en/stable/) is another framework that provides an abstraction layer over the chunking and vectorization layer.

This example aims at vectorizing documents at paragraph level defining chunks of size `1024` with an overlap of `32:

```python
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

reader = SimpleDirectoryReader(input_dir="data")
docs = reader.load_data()

index = VectorStoreIndex.from_documents(
    docs,
    transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=32)],
)
```


