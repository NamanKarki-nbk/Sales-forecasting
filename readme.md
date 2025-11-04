# 🛒 Intelligent Sales Forecasting & Query System

A hybrid AI system that combines **time-series forecasting** with
**natural language querying** to provide intelligent sales analytics and
business insights for retail operations in Nepal.

[![Python
3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Prophet](https://img.shields.io/badge/Prophet-Time%20Series-orange.svg)](https://facebook.github.io/prophet/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

------------------------------------------------------------------------

## 📋 Table of Contents

-   [Overview](#overview)
-   [Key Features](#key-features)
-   [System Architecture](#system-architecture)
-   [Tech Stack](#tech-stack)
-   [Installation](#installation)
-   [Quick Start](#quick-start)
-   [Project Structure](#project-structure)
-   [How It Works](#how-it-works)
-   [Usage Examples](#usage-examples)
-   [Data Requirements](#data-requirements)
-   [Configuration](#configuration)
-   [Contributing](#contributing)
-   [License](#license)
-   [Acknowledgments](#acknowledgments)

------------------------------------------------------------------------

## 🎯 Overview

This system enables business users to ask natural language questions
about sales data and receive accurate, context-aware responses. It
combines:

-   **📊 Sales Forecasting**: Prophet-based time-series forecasting for
    100+ weeks ahead
-   **💬 Natural Language Querying**: Ask questions in plain
    English/Nepali context
-   **🧠 RAG System**: Retrieval-Augmented Generation for contextual
    business insights
-   **🔀 Hybrid Intelligence**: Seamlessly combines numerical analysis
    with contextual explanations

**Example Queries:**

    "What will be the total sales for Electronics at Bhat-Bhateni in 2025?"
    "How do Dashain holidays affect sales?"
    "Show me the sales trend for Grocery department"
    "Compare sales between 2024 and 2025"

... (content truncated to fit tool limits) ...


## Architecture

``` mermaid
graph TB
    subgraph Input["📥 INPUT DATA"]
        SalesData[Sales Data CSV<br/>Historical Weekly Sales<br/>Store × Department]
        HolidayData[Holiday Knowledge Base<br/>nepal_holiday_kb<br/>Festival Information]
    end
    
    subgraph Forecasting["🔮 FORECASTING SYSTEM"]
        ProphetModel[Prophet Model<br/>Time Series Analysis]
        ModelCache[(Model Storage<br/>Pickle Files)]
        ForecastCache[(Forecast Storage<br/>CSV Files)]
        
        SalesData --> ProphetModel
        ProphetModel --> ModelCache
        ProphetModel --> ForecastCache
    end
    
    subgraph RAG["🤖 RAG SYSTEM"]
        Embeddings[HuggingFace Embeddings<br/>all-MiniLM-L6-v2]
        VectorStore[(FAISS Vector Store<br/>Semantic Search)]
        LLM[Ollama LLM<br/>Mistral Model]
        
        HolidayData --> Embeddings
        Embeddings --> VectorStore
        VectorStore --> LLM
    end
    
    subgraph Output["📊 OUTPUT & INTERFACE"]
        Predictions[Sales Predictions<br/>104 weeks forecast<br/>with confidence intervals]
        Insights[Holiday Insights<br/>Natural Language Answers<br/>Contextual Information]
        
        ForecastCache --> Predictions
        LLM --> Insights
    end
    
    subgraph Business["💼 BUSINESS VALUE"]
        Inventory[Inventory Planning<br/>Stock Optimization]
        Staffing[Staff Scheduling<br/>Resource Allocation]
        Budget[Budget Planning<br/>Revenue Forecasting]
        Marketing[Marketing Timing<br/>Promotional Campaigns]
        
        Predictions --> Inventory
        Predictions --> Staffing
        Predictions --> Budget
        Insights --> Marketing
    end
    
    %% User Interaction
    User([👤 Business User<br/>Store Manager])
    
    User -->|Request Forecast| Forecasting
    User -->|Ask Questions| RAG
    Predictions --> User
    Insights --> User
    User --> Business
    
    %% Styling
    classDef inputStyle fill:#E3F2FD,stroke:#1976D2,stroke-width:3px
    classDef processStyle fill:#FFF3E0,stroke:#F57C00,stroke-width:3px
    classDef outputStyle fill:#F3E5F5,stroke:#7B1FA2,stroke-width:3px
    classDef businessStyle fill:#E8F5E9,stroke:#388E3C,stroke-width:3px
    classDef userStyle fill:#FFE0B2,stroke:#E64A19,stroke-width:3px
    
    class Input inputStyle
    class Forecasting,RAG processStyle
    class Output outputStyle
    class Business businessStyle
    class User userStyle 
```

