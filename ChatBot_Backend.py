# ***************************************** IMPORTING THE DEPENDENCIES****************************************************

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph , START , END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage , HumanMessage , SystemMessage
from langgraph.graph.message import add_messages
from typing import Annotated , TypedDict , Dict , Optional
from langgraph.prebuilt import ToolNode , tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import requests
import os 
import tempfile
from langgraph.types import interrupt , Command
from langchain_core.runnables import RunnableConfig

# LOADING THE ENVIRONMENT VARIABLE 

load_dotenv ()

# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROQ_API_KEY_2")
)


# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS : Dict[str , any] = {}

_THREAD_METADATA : Dict[str, any] = {}

def _get_retriever(thread_id : Optional[str]):
    """ Fetch the retriever for a thread if available """
    if thread_id and thread_id in _THREAD_RETRIEVERS :
        return _THREAD_RETRIEVERS[thread_id]
    return None



def ingest_pdf(file_bytes : bytes , thread_id : str , filename :Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        return ValueError ("No bytes recieved for ingestion")
    with tempfile.NamedTemporaryFile(delete = False , suffix =".pdf") as temp_file :
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try :
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000 , 
            chunk_overlap= 200
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

        vectorstore = FAISS.from_documents(
        chunks,
        embeddings
)

        retriever = vectorstore.as_retriever(
            
                search_type = "similarity",
                search_kwargs = {'k':3}
            )
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            'filename' : filename or os.path.basename(temp_file),
            'documents' : len(docs),
            'chunks' : len(chunks)
        }
        return {
        'filename' : filename or os.path.basename(temp_path),
        'documents' : len(docs),
        'chunks' : len(chunks)
        }
    finally :
        try:
            os.remove(temp_path)
        except OSError:
            pass

    
# CREATING THE TOOL
search_tool = DuckDuckGoSearchRun(region = "us-en")

# CREATING THE CALCULATOR TOOL
@tool
def calculator(first_num: float , second_num : float , operation : str) -> dict:
    """ calculator tool which perform bassic aritmatic operations on two numbers.
    supported operations : add, mul , sub, div  """
    try :
        if operation == "add":
            result = first_num + second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "div":
            if second_num == 0:
                return {"error" : "Division by 0 in not allowed"}
            result = first_num / second_num
        else:
            return {"error" : f"Invalid operation {operation}"}
        return {
            "first_num": first_num,
            "second_num":second_num,
            "operation":operation,
            "result" : result
            }
    except Exception as e :
        return {"error" : str(e)}
import os

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


@tool 
def get_stock_price(symbol : str) -> dict:
    """ Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=API_KEY"
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol: str , quantity : int)  -> dict :
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """

    decision = interrupt( { "type": "stock_purchase_approval", "message": "Human approval required before purchasing stock.", "symbol": symbol, "quantity": quantity, "question": ( f"Do you approve buying {quantity} shares " f"of {symbol}?" ) } )
    if isinstance(decision ,str) and decision.strip().lower() == "yes":
        return {
            'status': 'approved',
            'symbol': symbol,
            'quantity' : quantity,
            'message' : f'purchase order placed for {quantity} and the {symbol} stock'
        }
    return {
        'status' :'denied',
        'message' : f'purchase order cancelled for {quantity} and the {symbol}',
        'quantity' : quantity,
        'symbol': symbol
    }
@tool
def rag_tool(
    query: str,
    config: RunnableConfig
) -> dict:
    """
    Retrieve relevant information from the PDF
    associated with the current chat thread.

    The thread_id is automatically taken from the
    LangGraph runtime config.
    """

    # ------------------------------------------------------------
    # GET CURRENT THREAD ID FROM LANGGRAPH CONFIG
    # ------------------------------------------------------------

    thread_id = (
        config
        .get("configurable", {})
        .get("thread_id")
    )

    # ------------------------------------------------------------
    # GET RETRIEVER FOR THIS THREAD
    # ------------------------------------------------------------

    retriever = _get_retriever(
        str(thread_id)
        if thread_id
        else None
    )

    # ------------------------------------------------------------
    # NO PDF FOUND
    # ------------------------------------------------------------

    if retriever is None:

        return {
            "error": (
                "No document indexed for this chat. "
                "Please upload a PDF first."
            ),
            "query": query,
            "thread_id": thread_id
        }

    # ------------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENT CHUNKS
    # ------------------------------------------------------------

    try:

        results = retriever.invoke(query)

        # Extract text from retrieved chunks
        context = [
            doc.page_content
            for doc in results
        ]

        # Extract metadata such as page numbers
        metadata = [
            doc.metadata
            for doc in results
        ]

        # --------------------------------------------------------
        # RETURN RETRIEVED CONTEXT
        # --------------------------------------------------------

        return {
            "query": query,
            "thread_id": thread_id,
            "context": context,
            "metadata": metadata
        }

    except Exception as e:

        return {
            "error": f"RAG retrieval failed: {str(e)}",
            "query": query,
            "thread_id": thread_id
        }

tools = [rag_tool,search_tool , get_stock_price , calculator, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


# -------------------
# 4. NODES
# -------------------


def chat_node(state : ChatState,config = None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")


        system_message = SystemMessage(
        content=(
        "You are a factual AI assistant.\n"
        "For questions about the uploaded PDF, call rag_tool once and answer only from its context.\n"
        "Do not call rag_tool again unless its result is empty or clearly insufficient.\n"
        "Never invent facts, numbers, quotes, or page references.\n"
        "If the PDF does not contain the answer, say: "
        "'The uploaded PDF does not provide this information.'\n"
        "Use search_tool for web/current information, calculator for math, "
        "get_stock_price for stock prices, and purchase_stock for purchases.\n"
        "A purchase is not completed until human approval is received."
    )
)
        
    

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}



tool_node = ToolNode(tools)

conn = sqlite3.connect(database="ChatBot.DB",check_same_thread= False)
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)

# ADDING THE NODES IN THE GRAPH 
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

# ADDING THE EDGES IN THE GRAPH
graph.add_edge(START , "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer = checkpointer)


# ********************************************************************
# 11. NORMAL CHAT INVOCATION
# ********************************************************************

def invoke_chat(
    user_input: str,
    thread_id: str
) -> dict:
    """
    Send a normal user message to the chatbot.

    Returns:

    Completed:
        {
            "status": "completed",
            "thread_id": "...",
            "message": "..."
        }

    HITL:
        {
            "status": "interrupt",
            "thread_id": "...",
            "approval": {...}
        }
    """

    thread_id = str(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = {
        "messages": [
            HumanMessage(
                content=user_input
            )
        ]
    }

    response = chatbot.invoke(
        state,
        config=config
    )

    # ------------------------------------------------------------
    # CHECK HITL INTERRUPT
    # ------------------------------------------------------------

    interrupts = response.get(
        "__interrupt__",
        []
    )

    if interrupts:

        interrupt_data = interrupts[0]

        return {
            "status": "interrupt",
            "thread_id": thread_id,
            "interrupt_id": getattr(
                interrupt_data,
                "id",
                None
            ),
            "approval": interrupt_data.value
        }

    # ------------------------------------------------------------
    # NORMAL COMPLETION
    # ------------------------------------------------------------

    messages = response.get(
        "messages",
        []
    )

    final_message = (
        messages[-1].content
        if messages
        else ""
    )

    return {
        "status": "completed",
        "thread_id": thread_id,
        "message": final_message
    }


# ********************************************************************
# 12. RESUME AFTER HUMAN APPROVAL
# ********************************************************************

def resume_chat(
    thread_id: str,
    decision: str
) -> dict:
    """
    Resume an interrupted graph execution.

    decision:
        yes
        no
    """

    thread_id = str(thread_id)

    decision = str(
        decision
    ).strip().lower()

    # ------------------------------------------------------------
    # VALIDATE DECISION
    # ------------------------------------------------------------

    if decision not in {"yes", "no"}:

        return {
            "status": "error",
            "thread_id": thread_id,
            "message": (
                "Invalid approval decision. "
                "Use 'yes' or 'no'."
            )
        }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # ------------------------------------------------------------
    # RESUME INTERRUPTED GRAPH
    # ------------------------------------------------------------

    response = chatbot.invoke(
        Command(
            resume=decision
        ),
        config=config
    )

    # ------------------------------------------------------------
    # CHECK FOR ANOTHER INTERRUPT
    # ------------------------------------------------------------

    interrupts = response.get(
        "__interrupt__",
        []
    )

    if interrupts:

        interrupt_data = interrupts[0]

        return {
            "status": "interrupt",
            "thread_id": thread_id,
            "interrupt_id": getattr(
                interrupt_data,
                "id",
                None
            ),
            "approval": interrupt_data.value
        }

    # ------------------------------------------------------------
    # COMPLETED
    # ------------------------------------------------------------

    messages = response.get(
        "messages",
        []
    )

    final_message = (
        messages[-1].content
        if messages
        else ""
    )

    return {
        "status": "completed",
        "thread_id": thread_id,
        "message": final_message
    }


# ********************************************************************
# 13. THREAD MANAGEMENT
# ********************************************************************

# def retrieve_all_threads():

#     all_threads = set()

#     for checkpoint in checkpointer.list(None):

#         thread_id = (
#             checkpoint.config
#             .get("configurable", {})
#             .get("thread_id")
#         )

#         if thread_id:
#             all_threads.add(
#                 thread_id
#             )

#     return list(all_threads)


def retrieve_all_threads():

    latest_by_thread = {}

    for checkpoint in checkpointer.list(None):

        config = checkpoint.config or {}

        configurable = config.get(
            "configurable",
            {}
        )

        thread_id = configurable.get(
            "thread_id"
        )

        if not thread_id:
            continue

        thread_id = str(thread_id)

        checkpoint_data = getattr(
            checkpoint,
            "checkpoint",
            {}
        )

        timestamp = checkpoint_data.get(
            "ts",
            ""
        )

        # Keep the latest checkpoint timestamp
        if (
            thread_id not in latest_by_thread
            or timestamp > latest_by_thread[thread_id]
        ):
            latest_by_thread[thread_id] = timestamp

    # Newest conversation first
    threads = sorted(
        latest_by_thread.keys(),
        key=lambda thread_id: latest_by_thread[thread_id],
        reverse=True
    )

    return threads

# ********************************************************************
# 14. PDF STATUS HELPERS
# ********************************************************************

def thread_has_document(
    thread_id: str
) -> bool:

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )


def thread_document_metadata(
    thread_id: str
) -> dict:

    return _THREAD_METADATA.get(
        str(thread_id),
        {}
    )


# ********************************************************************
# 15. OPTIONAL CLI TEST
# ********************************************************************

if __name__ == "__main__":

    print("=" * 60)
    print("Welcome to the HITL Agentic Chatbot")
    print("=" * 60)
    print("Type 'exit' to quit.")
    print()

    thread_id = "thread_1"

    while True:

        user_input = input("You: ")

        if user_input.lower().strip() in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "Exiting the chatbot. "
                "Thank you for using the chatbot."
            )

            break

        # --------------------------------------------------------
        # NORMAL CHAT
        # --------------------------------------------------------

        response = invoke_chat(
            user_input=user_input,
            thread_id=thread_id
        )

        # --------------------------------------------------------
        # HITL INTERRUPT
        # --------------------------------------------------------

        if response["status"] == "interrupt":

            approval = response["approval"]

            print()
            print("=" * 50)
            print("HUMAN APPROVAL REQUIRED")
            print("=" * 50)

            print(
                approval.get(
                    "question",
                    "Approval required."
                )
            )

            print(
                f"Symbol   : "
                f"{approval.get('symbol', 'N/A')}"
            )

            print(
                f"Quantity : "
                f"{approval.get('quantity', 'N/A')}"
            )

            print()

            decision = input(
                "Your response (yes/no): "
            ).strip().lower()

            # ----------------------------------------------------
            # RESUME GRAPH
            # ----------------------------------------------------

            response = resume_chat(
                thread_id=thread_id,
                decision=decision
            )

        # --------------------------------------------------------
        # FINAL RESPONSE
        # --------------------------------------------------------

        print()
        print(
            "Bot:",
            response.get(
                "message",
                ""
            )
        )

        print()