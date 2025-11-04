import os
from typing import TypedDict, Annotated, List, Literal
import operator
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# LangChain/Hugging Face Imports
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph Imports
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION & STATE ---

# Define the state for LangGraph
class AgentState(TypedDict):
    """Represents the state of our graph."""
    question: str
    answer: str
    context: Annotated[List[str], operator.add] 
    action: str # Used by the conditional edge in the graph

# --- 2. INITIALIZATION FUNCTIONS ---

# Configuration
VECTOR_DB_PATH = "./chroma_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"

# Hugging Face Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

def initialize_models_and_db():
    """Initializes LLM, Embeddings, and the Retriever."""
    print("--- Initializing HF Models & DB ---")
    
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. LLM Pipeline (Mistral 7B - requires significant resources)
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"⚠️ Could not load HuggingFace LLM: {e}. Ensure models are downloaded and hardware is sufficient.")
        # Fallback/placeholder LLM for flow testing if local model fails
        llm = None 

    # 3. Retriever (Load or Create)
    if not os.path.exists(VECTOR_DB_PATH):
        print("⚠️ ChromaDB not found. Creating knowledge base...")
        # Load and split documents
        all_docs = []
        if os.path.exists(KNOWLEDGE_BASE_DIR):
            for filename in os.listdir(KNOWLEDGE_BASE_DIR):
                if filename.endswith(".txt"):
                    loader = TextLoader(os.path.join(KNOWLEDGE_BASE_DIR, filename))
                    all_docs.extend(loader.load())

        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings, 
                persist_directory=VECTOR_DB_PATH
            )
            vectorstore.persist()
            print("✅ Knowledge Base created.")
        else:
            print(f"❌ No documents found in {KNOWLEDGE_BASE_DIR}. Retrieval will be non-functional.")
            return llm, None
    else:
        print("✅ ChromaDB loaded from disk.")
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=embeddings
        )
        
    retriever = vectorstore.as_retriever(k=3)
    return llm, retriever

# --- 3. LANgGRAPH NODES ---

def plan(state: AgentState, llm):
    """Node 1: Interprets the query and decides if retrieval is needed."""
    question = state["question"]
    print(f"\n--- 🧠 NODE: PLAN for query: '{question}' ---")
    
    prompt_str = f"""
    You are an expert planning agent. Analyze the user's question and decide the next step.
    If the question requires specialized knowledge (RAG), respond with only the word: RETRIEVE
    If the question is general or a greeting, respond with only the word: DIRECT_ANSWER

    Question: {question}
    Decision:"""
    
    planning_output = llm.invoke(prompt_str).strip().upper()
    action = "RETRIEVE" if "RETRIEVE" in planning_output else "DIRECT_ANSWER"
    
    print(f"📝 Plan determined: {action}")
    return {"action": action}

def retrieve(state: AgentState, retriever):
    """Node 2: Performs RAG using the vector database."""
    if not retriever:
        return {"context": ["Error: Retrieval skipped, DB not ready."]}
        
    question = state["question"]
    print(f"\n--- 🔍 NODE: RETRIEVE context for query: '{question}' ---")
    
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    
    print(f"📄 Retrieved {len(context)} documents.")
    return {"context": context}

def answer(state: AgentState, llm):
    """Node 3: Generates the final answer using the LLM and retrieved context."""
    question = state["question"]
    context = "\n---\n".join(state["context"])
    print(f"\n--- 💡 NODE: ANSWER generation. Context length: {len(context)} ---")

    if "Error: Retrieval skipped" in context or not context:
        template = (
            "You are a helpful Q&A assistant. Answer the following question based ONLY on your internal knowledge. "
            "Question: {question}"
        )
    else:
        template = (
            "You are a helpful Q&A assistant. Use the following retrieved context to answer the user's question. "
            "If the context does not contain the answer, state that you cannot find the relevant information "
            "in the provided documents.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {question}"
        )

    prompt = ChatPromptTemplate.from_template(template)
    answer_chain = prompt | llm | StrOutputParser()
    final_answer = answer_chain.invoke({"context": context, "question": question})
    
    print("✅ Final Answer Generated.")
    return {"answer": final_answer}

def reflect(state: AgentState, llm):
    """Node 4: Evaluates the generated answer for relevance and completeness."""
    question = state["question"]
    answer = state["answer"]
    
    print("\n--- 🧐 NODE: REFLECT on the generated answer ---")
    
    reflection_prompt_str = f"""
    You are a critical reflection agent. Review the following.
    If the Answer is directly relevant to the Question, respond with only the word: ACCEPT
    If the Answer is irrelevant, incomplete, or poor, respond with only the word: REVISE

    Question: {question}
    Answer: {answer}
    Reflection:"""

    reflection_result = llm.invoke(reflection_prompt_str).strip().upper()
    reflection_status = "ACCEPT" if "ACCEPT" in reflection_result else "REVISE"

    print(f"🔬 Reflection result: {reflection_status}")
    return {"reflection_status": reflection_status}

# --- 4. LANgGRAPH DEFINITION ---

def create_rag_agent(llm, retriever):
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Use functools.partial to pass initialized objects (llm, retriever) to the nodes
    workflow.add_node("plan", lambda state: plan(state, llm))
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("answer", lambda state: answer(state, llm))
    workflow.add_node("reflect", lambda state: reflect(state, llm))

    workflow.set_entry_point("plan")

    def route_plan(state):
        """Route question based on plan's decision."""
        if state["action"] == "RETRIEVE":
            return "retrieve"
        else:
            return "answer"

    workflow.add_conditional_edges("plan", route_plan, {"retrieve": "retrieve", "answer": "answer"})
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", "reflect")
    workflow.add_edge("reflect", END)

    app = workflow.compile()
    print("\n--- ✅ Agent Workflow Compiled ---")
    return app

# --- 5. MAIN EXECUTION ENTRY (For Streamlit) ---

def get_agent_instance():
    """Entry function to get the compiled agent."""
    load_dotenv()
    llm, retriever = initialize_models_and_db()
    if llm and retriever:
        return create_rag_agent(llm, retriever)
    elif llm:
        print("Agent is running without RAG (no documents found).")
        # Create agent without RAG, plan will always hit 'answer'
        return create_rag_agent(llm, None)
    else:
        print("Agent failed to initialize LLM. Cannot run.")
        return None

if __name__ == '__main__':
    # This block is for command-line testing/DB setup only
    llm, retriever = initialize_models_and_db()
    if llm and retriever:
        app = create_rag_agent(llm, retriever)
        # Example run
        initial_state = {"question": "What are the benefits of renewable energy?", "answer": "", "context": [], "action": ""}
        final_state = app.invoke(initial_state)
        print("\n--- TEST RUN RESULT ---")
        print(f"Answer: {final_state['answer']}")