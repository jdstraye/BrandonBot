# BrandonBot Project

## Overview
BrandonBot is an open-source, RAG-based AI chatbot for a political candidate, designed to answer questions about Brandon's political positions with a 5-tier confidence system and zero API costs. The project aims to provide transparent, accurate information reflecting Brandon's stated views, leveraging marketing principles for communication style. It is a Replit-native application emphasizing open-source components and local inference.

## User Preferences
I prefer iterative development with clear, concise communication. When making changes, please explain the "why" behind them, not just the "what." I value performance and cost-efficiency. Do not make changes to the `replit.md` file without explicit instruction.

## System Architecture
The system utilizes a FastAPI backend, an embedded Weaviate vector database, and a CPU-optimized Phi-3 Mini ONNX model for local inference. It employs Sentence-Transformers for embeddings and SQLite for interaction logging. The frontend is built with vanilla HTML/CSS/JS.

**Key Architectural Decisions & Features:**
-   **4-Stage Multi-Dimensional RAG Pipeline**: Includes Question Analysis, Multi-Source Retrieval, Enhanced Response Generation, and Response Delivery, adapting based on context and confidence.
-   **Retrieval-First Architecture**: Ensures consistent confidence evaluation by routing all questions through a retrieve → build context → Phi-3 generation path.
-   **RAG Pipeline Refactored**: Separates content search (BrandonPlatform, PreviousQA, PartyPlatform) from style guidance (MarketGurus), applying style only when content confidence is high.
-   **Smart Boundary-Aware Chunking**: Optimizes document ingestion.
-   **Trust Multiplier System**: Calculates confidence as `Similarity × Trust Factor`.
-   **3-Tier Knowledge Base (Weaviate Collections)**: Includes BrandonPlatform (1.0x trust), PreviousQA (1.0x trust), and PartyPlatform (0.6x trust).
-   **Marketing-Guided Communication**: Influences communication style using copywriting wisdom from the MarketGurus collection, adapted to question analysis and prospect awareness.
-   **Character-Breaking Behavior**: The bot maintains character for high-confidence policy questions but can break character for comparisons, recent events, or low-confidence responses.
-   **Question Type Detection**: Identifies and frames responses for 6 types: comparison, statistics, truth-seeking, recent_event, policy, and low_confidence.
-   **Callback System**: Offers personal callbacks for low-confidence answers and external search scenarios.
-   **Privacy-First**: Implements opt-in logging for ethical data collection.
-   **CPU-Optimized**: Designed for CPU-only environments using INT4 quantized models.

## External Dependencies
-   **FastAPI**: Python web framework.
-   **Weaviate Embedded**: Local, embedded vector database.
-   **Phi-3 Mini ONNX Runtime**: Local large language model.
-   **Sentence-Transformers**: For generating embeddings.
-   **SQLite**: For logging interactions.
-   **DuckDuckGo Web Search**: For external search capabilities.
-   **OpenAI API (Planned)**: Future integration for enhanced performance and quality.