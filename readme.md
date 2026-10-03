# 🤖 Agentic Nexus

### A Multi-Tool Agentic AI Assistant built with LangGraph, RAG and Human-in-the-Loop

Agentic Nexus is an **agentic AI chatbot** that can reason about a user's request and decide when to use different tools instead of behaving like a simple question-answering chatbot.

The project combines:

- LangGraph for agent orchestration
- Groq LLM for reasoning
- RAG for querying uploaded PDF documents
- Web search for external information
- Calculator for arithmetic operations
- Alpha Vantage for stock prices
- Human-in-the-Loop approval for stock purchases
- SQLite checkpointing for conversation memory
- Streamlit for the user interface
- Live tool execution visibility in the frontend

The main goal of the project is to demonstrate how an AI agent can **decide, use tools, pause for human approval, and continue execution safely**.

---

# 📌 Why I Built This

Traditional chatbots mainly follow this pattern:

User → LLM → Answer

That approach becomes insufficient when the system needs to:

- retrieve information from user documents
- search the web
- perform calculations
- access external APIs
- perform actions that require approval
- maintain separate conversations
- recover from interrupted workflows

Agentic Nexus changes the workflow to:

User → Agent → Tool Selection → Tool Execution → Result → Agent → Final Response

For sensitive actions, the workflow becomes:

User → Agent → Action → Human Approval → Continue/Cancel

---

# 🧠 What Problem Does It Solve?

The project focuses on three major problems.

## 1. LLMs do not always have access to private user information

A general LLM cannot automatically know the contents of a PDF uploaded by a user.

Agentic Nexus solves this using **Retrieval-Augmented Generation (RAG)**.

The uploaded PDF is:

1. Loaded using PyPDFLoader
2. Split into smaller chunks
3. Converted into embeddings
4. Stored in a FAISS vector store
5. Retrieved when the user asks a document-related question

The relevant chunks are then provided to the LLM so it can answer from the document.

---

## 2. AI agents sometimes need external tools

Instead of forcing the LLM to answer everything by itself, the agent can use specialized tools.

Currently available tools:

| Tool | Purpose |
|------|---------|
| `rag_tool` | Retrieve information from uploaded PDFs |
| `search_tool` | Search the web |
| `calculator` | Perform arithmetic |
| `get_stock_price` | Retrieve stock market data |
| `purchase_stock` | Simulate a stock purchase with human approval |

The LLM decides which tool is appropriate for the user's request.

---

## 3. Some AI actions should not happen without human approval

A major feature of this project is **Human-in-the-Loop (HITL)**.

For example:

```text
User:
Buy 10 shares of AAPL

The agent can decide to call:

purchase_stock()

But the tool does not immediately complete the action.

Instead:

purchase_stock()
      ↓
interrupt()
      ↓
Human approval required
      ↓
Approve / Deny
      ↓
Command(resume="yes/no")
      ↓
Workflow continues

This creates an explicit human checkpoint before the action is completed.

🏗️ System Architecture
                         ┌──────────────────────┐
                         │      Streamlit UI    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     LangGraph Agent  │
                         │      (chat_node)     │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
               RAG Tool        Web Search       Calculator
                    │
                    ▼
                FAISS
                    │
                    ▼
              Uploaded PDF

                    ┌───────────────────────┐
                    │    Stock Price Tool   │
                    └───────────┬───────────┘
                                │
                                ▼
                         Alpha Vantage API


                    ┌───────────────────────┐
                    │   Purchase Stock Tool │
                    └───────────┬───────────┘
                                │
                                ▼
                         Human Approval
                          ↙           ↘
                       YES             NO
                        │               │
                        ▼               ▼
                    Continue         Cancel
🔄 How the Agent Works
Normal question

Example:

What is the meaning of assets according to the uploaded PDF?

Workflow:

User
 ↓
LangGraph
 ↓
LLM decides RAG is required
 ↓
rag_tool
 ↓
FAISS similarity search
 ↓
Relevant PDF chunks
 ↓
LLM
 ↓
Answer
🌐 Web Search Example

For a request requiring external information:

Search the web for the latest information about X

Workflow:

User
 ↓
LLM
 ↓
search_tool
 ↓
DuckDuckGo
 ↓
Search results
 ↓
LLM
 ↓
Answer
🧮 Calculator Example
What is 154 * 38?

Workflow:

User
 ↓
LLM
 ↓
calculator
 ↓
Result
 ↓
LLM
 ↓
Answer
📈 Stock Price Example
What is the current price of AAPL?

Workflow:

User
 ↓
LLM
 ↓
get_stock_price
 ↓
Alpha Vantage API
 ↓
Result
 ↓
LLM
 ↓
Answer
🛑 Human-in-the-Loop Example

For a purchase request:

User:
Buy 10 shares of AAPL

The agent calls:

purchase_stock

The tool pauses execution:

⏸ Human approval required

The Streamlit UI displays:

Do you approve buying 10 shares of AAPL?

[ ✅ Approve Purchase ]
[ ❌ Deny Purchase ]

If approved:

Command(resume="yes")

If denied:

Command(resume="no")

The same LangGraph thread is resumed after the decision.

🧵 Thread-Based Conversations

Every conversation receives its own unique thread ID.

Example:

Thread A
    └── Conversation 1

Thread B
    └── Conversation 2

Thread C
    └── Conversation 3

The Streamlit sidebar allows the user to switch between conversations.

The current active thread is clearly displayed in the UI.

SQLite checkpointing is used to preserve LangGraph conversation state.

🔎 Live Agent Activity

The frontend also exposes the internal tool workflow.

Instead of only showing:

Bot: 40

the user can see:

🤔 Agent is thinking...

🔧 Tool requested: calculator

↳ Arguments:
{'first_num': 25, 'second_num': 40, 'operation': 'mul'}

⚙️ Executing: calculator

✅ Tool finished: calculator

🤔 Agent is processing the tool result...

🤖 Agent generated a response.

This makes the agent's execution more transparent and easier to debug.

🧱 Tech Stack
Technology	Purpose
Python	Application development
LangGraph	Agent workflow orchestration
LangChain	LLM and tool integration
Groq	LLM inference
Streamlit	Frontend
FAISS	Vector similarity search
HuggingFace Embeddings	Document embeddings
PyPDF	PDF processing
DuckDuckGo	Web search
Alpha Vantage	Stock market API
SQLite	Conversation checkpointing
🛠️ Problems I Faced While Building This Project

This project was built iteratively, and several real implementation problems appeared during development.

1. PDF Retrieval Was Not Reaching the Correct Thread

Initially, the RAG tool depended on the LLM passing the correct thread_id.

That caused reliability problems because tool arguments are generated by the model.

Problem

The model could call:

rag_tool(query)

without providing the correct thread ID.

This resulted in the application failing to find the retriever associated with the current conversation.

Solution

The RAG tool was changed to receive the LangGraph runtime configuration:

def rag_tool(
    query: str,
    config: RunnableConfig
):

The thread ID is then obtained directly from:

config["configurable"]["thread_id"]

This removes the responsibility from the LLM to generate or remember the thread ID.

2. RAG Tool Returned Incorrectly

An early implementation contained:

return

{
    "query": query,
    "context": context
}

Because of the newline after return, Python returned None.

Solution

The dictionary was returned directly:

return {
    "query": query,
    "context": context,
    "metadata": metadata
}


3. Human-in-the-Loop Required Persistent Graph State

A normal function call is not enough for HITL because execution needs to pause and later continue from the same point.

Solution

LangGraph checkpointing was added with SQLite:

SqliteSaver

and the purchase tool uses:

interrupt()

The workflow is resumed using:

Command(resume="yes")

or:

Command(resume="no")

This allows an interrupted agent execution to continue from the saved state.

4. Tool Execution Was Invisible to the User

Initially, the chatbot only displayed the final answer.

That made it difficult to understand what the agent was doing.

Solution

The frontend was changed to stream LangGraph updates and display:

🤔 Agent thinking
🔧 Tool requested
⚙️ Executing
✅ Tool finished

The user can therefore observe the agent's tool workflow in real time.

5. Large RAG Context Caused Model Token Errors

During testing, repeated RAG calls caused a Groq error similar to:

Request too large for model
Requested 8617
Limit 8000 TPM

The agent had called the RAG tool multiple times and the retrieved context increased the total request size.

Solution

The agent instructions were tightened to:

call RAG for PDF questions
avoid unnecessary repeated RAG calls
answer only from retrieved context
avoid inventing facts
clearly state when the PDF does not contain the requested information

The number of retrieved chunks was also reduced to control context size.

6. Hallucination Testing

The chatbot was tested with questions asking for very specific facts that may not exist in the uploaded PDF.

Example:

According to the uploaded PDF, what was the exact monthly salary
of Robert's biological father in 1975?

The goal of these tests was to ensure that the system does not fabricate an answer when the retrieved document does not contain the required information.

The agent prompt was therefore designed to explicitly distinguish between:

Known from retrieved context

and:

Not available in the document
⚠️ Current Limitations

This version intentionally has some limitations that are planned for future iterations.

PDF index persistence

Currently, the FAISS retriever is kept in application memory.

Therefore:

Upload PDF
   ↓
FAISS index created
   ↓
Retriever stored in memory

If the Python/Streamlit process is restarted, the in-memory retriever is lost.

The next planned improvement is to persist FAISS indexes to disk or another persistent vector database so that:

Upload PDF today
        ↓
Shutdown PC
        ↓
Start PC after several days
        ↓
Open same thread
        ↓
PDF remains available

This is intentionally documented as a future improvement rather than treated as an implemented feature.

🚀 Future Improvements

Planned improvements include:

Persistent FAISS indexes per thread
Persistent document metadata
Better PDF lifecycle management
More granular tool permissions
More robust HITL approval workflows
Tool error recovery
Better conversation naming
Production database instead of local SQLite
Deployment with persistent storage
Authentication and user-specific document isolation.


⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/agentic-nexus.git

Enter the project directory:

cd agentic-nexus

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Create the environment file:

.env

Add your API keys.

▶️ Running the Application

Start the Streamlit application:

streamlit run ChatBot_Frontend.py

The browser will open the Streamlit interface.

🧪 Example Prompts
PDF / RAG
What does the book say about assets and liabilities?
Web Search
Search the web for the latest information about NVIDIA.
Calculator
Calculate 1250 * 17.
Stock Price
What is the latest price of AAPL?
Human-in-the-Loop
Buy 10 shares of AAPL.

The purchase request should pause and require human approval.

🎯 What This Project Demonstrates

This project demonstrates practical implementation of:

Agentic AI
LangGraph stateful workflows
Tool calling
Retrieval-Augmented Generation
Vector similarity search
Human-in-the-Loop systems
API integration
SQLite checkpointing
Multi-thread conversations
Streaming agent execution
Hallucination-resistant prompting
Debugging and iterative system design

The focus is not only on generating answers, but on building an AI system that can reason about when to retrieve information, when to call a tool, when to ask for human approval, and how to continue after an interruption.

👨‍💻 Author

Built as an Agentic AI portfolio project to explore production-oriented LangGraph architectures.

⭐ Project Highlights
LLM
 ↓
Agent
 ↓
Tool Selection
 ├── RAG
 ├── Web Search
 ├── Calculator
 ├── Stock Price
 └── Stock Purchase
          ↓
     Human Approval

Agentic Nexus is designed around a simple principle:

AI should automate what it can, retrieve what it doesn't know, and ask a human before taking sensitive actions.



THANK YOU.......