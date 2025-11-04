

# 🧠 LangGraph RAG Agent with Dynamic Self-Correction

A robust, modular AI Agent built on the **LangGraph** framework to perform **Retrieval-Augmented Generation (RAG)** using local, open-source **Hugging Face** models, deployed via a **Streamlit** user interface. This project demonstrates advanced agentic workflow design, dynamic tool use, and basic self-correction logic.

-----

## ✨ Project Highlights and Best Features

### 🚀 Technical Excellence

  * **Advanced Agent Orchestration (LangGraph):** Implements a stateful, cyclic workflow with four distinct nodes (`plan`, `retrieve`, `answer`, `reflect`), demonstrating expertise in complex agent architecture beyond simple sequential chains.
  * **Decoupled RAG Pipeline (Hugging Face & ChromaDB):** Successfully integrates entirely open-source components for the RAG stack, including **Mistral-7B-Instruct-v0.2** (LLM) and **BAAI/bge-small-en-v1.5** (Embeddings), ensuring full control and local execution.
  * **Dynamic Decision-Making (`plan` node):** The agent intelligently determines whether to execute the expensive retrieval step based on the query, optimizing performance and resource consumption.

### 🌟 Key Features

| Feature | Description | Technical Implementation |
| :--- | :--- | :--- |
| **Self-Correction Loop** | The **`reflect`** node acts as an LLM-as-a-Judge, validating the generated answer against the original query to ensure relevance and completeness before final output. | Constrained string output parsing within LangGraph flow. |
| **Interactive Deployment** | Provides a seamless, web-based chat experience for real-time testing and demonstration of the agent's capabilities. | **Streamlit** application (`app.py`) for the frontend interface. |
| **Modular Codebase** | The agent logic (`rag_agent_hf.py`) is decoupled from the UI (`app.py`), allowing for easy testing, refactoring, and integration into different applications. | Python modularity using `functools.partial` to pass dependencies. |

-----

## ⚙️ Agent Workflow: The Four-Node Graph

The agent processes a user query through the following stateful LangGraph stages:

1.  **`plan` (Decision):** Analyzes the user's `question`. Outputs **`RETRIEVE`** for domain-specific knowledge or **`DIRECT_ANSWER`** for general queries, routing the execution path.
2.  **`retrieve` (RAG):** If `RETRIEVE` is triggered, it queries the **ChromaDB** vector store using Hugging Face embeddings to fetch relevant `context` documents.
3.  **`answer` (Generation):** Uses the Mistral LLM to synthesize the final response, relying on the retrieved `context` (if available) or the LLM's internal knowledge (if retrieval was skipped).
4.  **`reflect` (Validation):** Evaluates the final `answer` against the initial `question`. Logs the status as **`ACCEPT`** or **`REVISE`**. The process terminates upon reflection.

-----

## ⚠️ Technical Challenges & Solutions

| Challenge | Description | Solution Implemented |
| :--- | :--- | :--- |
| **LLM Output Reliability** | Open-source LLMs (like Mistral) often struggle with reliable **JSON Structured Output** required for deterministic graph routing (`plan` and `reflect`). | Replaced Pydantic/JSON parsing with a **constrained string output prompt** ("Respond with ONLY the word: RETRIEVE"). This increased reliability for graph edges. |
| **Resource Management** | Running a 7B parameter LLM locally for the **`answer`** and **`reflect`** nodes demands significant CPU/VRAM, impacting initialization time and inference speed. | Utilized **`torch_dtype=torch.bfloat16`** and **`device_map="auto"`** in the Hugging Face pipeline to optimize model loading and memory usage across available resources. |
| **Dependency Management** | Synchronizing the dependencies for `transformers`, `torch`, `langchain-community`, and `chromadb` to run seamlessly in a single environment. | Provided a meticulously organized `requirements.txt` list, ensuring all necessary deep-learning and LangChain components are correctly installed. |

-----

## 📈 Future Improvements and Roadmap

The following enhancements are planned to further professionalize and optimize the agent:

1.  **Iterative Self-Correction:** Instead of terminating after `reflect`, modify the LangGraph edge to loop back to the **`answer`** node if the reflection status is **`REVISE`**, allowing for a **retry mechanism** with an improved prompt based on the reflection feedback.
2.  **RAG Evaluation Integration:** Incorporate **RAGAs** or a similar evaluation framework to programmatically assess RAG metrics (e.g., faithfulness, answer relevance) and log these results to a platform like **LangSmith** or **TruLens**.
3.  **Model Performance Optimization:** Explore quantization techniques (e.g., **Bitsandbytes 4-bit**) to further reduce the memory footprint of the Mistral model, enabling deployment on systems with fewer resources.
4.  **Asynchronous Execution:** Convert the LangGraph workflow to use asynchronous calls (`app.ainvoke()`) to improve the responsiveness and throughput of the Streamlit application.

-----

## 🚀 Setup and Execution

### 1\. Prerequisites

  * Python 3.9+
  * Sufficient RAM/VRAM (required for Mistral 7B).

### 2\. Installation

```bash
# Clone the repository
git clone [your-repo-link]
cd rag-langgraph-agent

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (requires large downloads)
pip install -r requirements.txt
```

### 3\. Knowledge Base Setup

1.  Create a directory: `mkdir knowledge_base`
2.  Add your source documents (e.g., `renewable_energy.txt`) into the `knowledge_base/` folder.

### 4\. Running the Streamlit Application

```bash
# Start the web interface
streamlit run app.py
```

The application will launch in your browser, automatically initializing the Hugging Face models and creating the persistent **ChromaDB** vector store on the first run.
