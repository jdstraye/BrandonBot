# BrandonBot Project

## Overview
BrandonBot is an open-source, RAG-based AI chatbot for a political candidate, designed to answer questions about Brandon's political positions. Its core features include a 5-tier confidence system, zero API costs, and a commitment to transparent, accurate information reflecting Brandon's stated views, all while leveraging marketing principles for effective communication. The project is a Replit-native application, emphasizing open-source components and local inference for cost-efficiency and performance.

## User Preferences
I prefer iterative development with clear, concise communication. When making changes, please explain the "why" behind them, not just the "what." I value performance and cost-efficiency. Do not make changes to the `replit.md` file without explicit instruction.

## System Architecture
The system employs a FastAPI backend, an embedded Weaviate vector database, and a CPU-optimized Phi-3 Mini ONNX model for local inference. Embeddings are generated using Sentence-Transformers, and interaction logs are stored in SQLite. The frontend is built with vanilla HTML/CSS/JS.

**Key Architectural Decisions & Features:**
-   **4-Stage Multi-Dimensional RAG Pipeline**: Dynamically adapts based on context and confidence, encompassing Question Analysis, Multi-Source Retrieval, Enhanced Response Generation, and Response Delivery.
-   **Retrieval-First Architecture**: Ensures consistent confidence evaluation by routing all questions through a retrieve → build context → Phi-3 generation path.
-   **RAG Pipeline Refactored**: Separates content search (BrandonPlatform, PreviousQA, PartyPlatform) from style guidance (MarketGurus), applying style only when content confidence is high.
-   **Smart Boundary-Aware Chunking**: Optimizes document ingestion for efficient retrieval.
-   **Trust Multiplier System**: Calculates response confidence as `Similarity × Trust Factor`.
-   **3-Tier Knowledge Base (Weaviate Collections)**: Organizes information into BrandonPlatform (1.0x trust), PreviousQA (1.0x trust), and PartyPlatform (0.6x trust).
-   **Marketing-Guided Communication**: Influences communication style using copywriting principles from the MarketGurus collection, adapted to question analysis and prospect awareness levels.
-   **Character-Breaking Behavior**: The bot maintains character for high-confidence policy questions but can break character for specific scenarios like comparisons, recent events, or low-confidence responses.
-   **Question Type Detection**: Identifies and frames responses for six types: comparison, statistics, truth-seeking, recent_event, policy, and low_confidence.
-   **Callback System**: Offers personal callbacks for low-confidence answers and scenarios requiring external search.
-   **Privacy-First**: Implements opt-in logging for ethical data collection.
-   **CPU-Optimized**: Designed to run efficiently in CPU-only environments using INT4 quantized models.

## External Dependencies
-   **FastAPI**: Python web framework for the backend.
-   **Weaviate Embedded**: Local, embedded vector database for knowledge storage.
-   **Phi-3 Mini ONNX Runtime**: Local large language model for inference.
-   **Sentence-Transformers**: Used for generating text embeddings.
-   **SQLite**: Database for logging user interactions.
-   **DuckDuckGo Web Search**: Integrated for external search capabilities.
-   **OpenAI API (Planned)**: Future integration for enhanced performance and quality.