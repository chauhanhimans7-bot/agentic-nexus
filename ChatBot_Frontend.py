import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

from ChatBot_Backend import (
chatbot,
    retrieve_all_threads,
    ingest_pdf,
    thread_document_metadata,
    resume_chat,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="🤖 Agentic Nexus",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_thread_id() -> str:
    """Generate a new unique thread ID."""
    return str(uuid.uuid4())


def add_thread(thread_id: str):
    """Add a thread to the sidebar list if not already present."""

    thread_id = str(thread_id)

    if thread_id not in st.session_state["Chat_thread"]:
        st.session_state["Chat_thread"].append(thread_id)


def reset_chat():
    """Start a fresh conversation."""

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["pending_approval"] = None

    add_thread(thread_id)


def load_conversation(thread_id: str):
    """Load saved messages from LangGraph checkpoint."""

    try:

        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": str(thread_id)
                }
            }
        )

        messages = state.values.get(
            "messages",
            []
        )

        formatted_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):

                if msg.content:
                    formatted_messages.append(
                        {
                            "role": "user",
                            "content": msg.content
                        }
                    )

            elif isinstance(msg, AIMessage):

                if msg.content:
                    formatted_messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content
                        }
                    )

        return formatted_messages

    except Exception as e:

        st.error(
            f"Failed to load conversation: {e}"
        )

        return []


# ============================================================
# LIVE ACTIVITY HELPER
# ============================================================

def render_activity(
    activity_placeholder,
    activity_log
):
    """Render live agent/tool activity."""

    if not activity_log:
        return

    activity_placeholder.markdown(
        "\n\n".join(activity_log)
    )


# ============================================================
# RUN CHAT WITH LIVE TOOL UPDATES
# ============================================================

def run_chat_with_live_updates(
    user_input: str,
    thread_id: str,
    activity_placeholder
):
    """
    Run LangGraph and display live:
    - agent thinking
    - tool requested
    - tool arguments
    - tool execution
    - tool completion
    - HITL interrupt
    """

    config = {
        "configurable": {
            "thread_id": str(thread_id)
        },
        "metadata": {
            "thread_id": str(thread_id)
        },
        "run_name": "chat_turn"
    }

    activity_log = []

    final_ai_message = ""

    interrupt_found = None

    # --------------------------------------------------------
    # INITIAL STATUS
    # --------------------------------------------------------

    activity_log.append(
        "🤔 **Agent is thinking...**"
    )

    render_activity(
        activity_placeholder,
        activity_log
    )

    try:

        # ----------------------------------------------------
        # STREAM LANGGRAPH
        # ----------------------------------------------------

        for update in chatbot.stream(
            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            },
            config=config,
            stream_mode="updates"
        ):

            # =================================================
            # INTERRUPT
            # =================================================

            if "__interrupt__" in update:

                interrupts = update[
                    "__interrupt__"
                ]

                if interrupts:

                    interrupt_found = interrupts[0]

                    activity_log.append(
                        "⏸️ **Human approval required**"
                    )

                    render_activity(
                        activity_placeholder,
                        activity_log
                    )

                continue

            # =================================================
            # CHAT NODE
            # =================================================

            if "chat_node" in update:

                node_data = update[
                    "chat_node"
                ]

                messages = node_data.get(
                    "messages",
                    []
                )

                for message in messages:

                    if isinstance(
                        message,
                        AIMessage
                    ):

                        # ------------------------------------
                        # TOOL REQUEST
                        # ------------------------------------

                        if message.tool_calls:

                            for tool_call in message.tool_calls:

                                tool_name = tool_call.get(
                                    "name",
                                    "unknown_tool"
                                )

                                tool_args = tool_call.get(
                                    "args",
                                    {}
                                )

                                activity_log.append(
                                    f"🔧 **Tool requested:** "
                                    f"`{tool_name}`"
                                )

                                if tool_args:

                                    activity_log.append(
                                        f"   ↳ Arguments: "
                                        f"`{tool_args}`"
                                    )

                                activity_log.append(
                                    f"⚙️ **Executing:** "
                                    f"`{tool_name}`"
                                )

                                render_activity(
                                    activity_placeholder,
                                    activity_log
                                )

                        # ------------------------------------
                        # AI RESPONSE
                        # ------------------------------------

                        elif message.content:

                            final_ai_message = (
                                message.content
                            )

                            activity_log.append(
                                "🤖 **Agent generated a response.**"
                            )

                            render_activity(
                                activity_placeholder,
                                activity_log
                            )

            # =================================================
            # TOOLS NODE
            # =================================================

            if "tools" in update:

                node_data = update[
                    "tools"
                ]

                messages = node_data.get(
                    "messages",
                    []
                )

                for message in messages:

                    if isinstance(
                        message,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message,
                            "name",
                            "tool"
                        )

                        activity_log.append(
                            f"✅ **Tool finished:** "
                            f"`{tool_name}`"
                        )

                        render_activity(
                            activity_placeholder,
                            activity_log
                        )

                activity_log.append(
                    "🤔 **Agent is processing the tool result...**"
                )

                render_activity(
                    activity_placeholder,
                    activity_log
                )

        # ======================================================
        # CHECK GRAPH STATE FOR INTERRUPT
        # ======================================================

        if interrupt_found is None:

            try:

                snapshot = chatbot.get_state(
                    config=config
                )

                if snapshot.tasks:

                    for task in snapshot.tasks:

                        task_interrupts = getattr(
                            task,
                            "interrupts",
                            None
                        )

                        if task_interrupts:

                            interrupt_found = (
                                task_interrupts[0]
                            )

                            activity_log.append(
                                "⏸️ **Human approval required**"
                            )

                            render_activity(
                                activity_placeholder,
                                activity_log
                            )

                            break

            except Exception:

                pass

        # ======================================================
        # RETURN INTERRUPT
        # ======================================================

        if interrupt_found is not None:

            return {
                "status": "interrupt",
                "approval": interrupt_found.value,
                "activity": activity_log
            }

        # ======================================================
        # RETURN COMPLETED
        # ======================================================

        return {
            "status": "completed",
            "message": final_ai_message,
            "activity": activity_log
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
            "activity": activity_log
        }


# ============================================================
# SESSION STATE
# ============================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "Chat_thread" not in st.session_state:
    st.session_state["Chat_thread"] = []


if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}


if "pending_approval" not in st.session_state:
    st.session_state["pending_approval"] = None


# ============================================================
# REFRESH THREAD LIST
# ============================================================

current_thread = str(
    st.session_state["thread_id"]
)

saved_threads = retrieve_all_threads()

# Remove duplicates while preserving backend order
threads = list(
    dict.fromkeys(saved_threads)
)

# Current new thread may not have a checkpoint yet
if current_thread not in threads:
    threads.insert(
        0,
        current_thread
    )

st.session_state["Chat_thread"] = threads


# ============================================================
# CURRENT THREAD
# ============================================================

thread_key = str(
    st.session_state["thread_id"]
)

add_thread(thread_key)

thread_docs = st.session_state[
    "ingested_docs"
].setdefault(
    thread_key,
    {}
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Rag Tool")


# ------------------------------------------------------------
# CURRENT CHAT
# ------------------------------------------------------------

st.sidebar.subheader(
    "🟢 Current Chat"
)

st.sidebar.markdown(
    f"**Thread ID**"
)

st.sidebar.code(
    thread_key,
    language=None
)


# ------------------------------------------------------------
# NEW CHAT
# ------------------------------------------------------------

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()
    st.rerun()


# ============================================================
# PDF STATUS
# ============================================================

if thread_docs:

    latest_doc = list(
        thread_docs.values()
    )[-1]

    st.sidebar.success(
        f"Using '{latest_doc.get('filename')}' "
        f"({latest_doc.get('chunks')} chunks "
        f"from {latest_doc.get('documents')} pages)"
    )

else:

    st.sidebar.info(
        "No documents uploaded yet."
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"]
)


if uploaded_pdf:

    if uploaded_pdf.name in thread_docs:

        st.sidebar.info(
            f"`{uploaded_pdf.name}` already processed "
            f"for this chat."
        )

    else:

        with st.sidebar.status(
            "Indexing PDF...",
            expanded=True
        ) as status_box:

            try:

                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name
                )

                thread_docs[
                    uploaded_pdf.name
                ] = summary

                status_box.update(
                    label="✅ PDF indexed",
                    state="complete",
                    expanded=False
                )

            except Exception as e:

                status_box.update(
                    label="❌ PDF indexing failed",
                    state="error",
                    expanded=True
                )

                st.sidebar.error(
                    str(e)
                )


# ============================================================
# PAST CONVERSATIONS
# ============================================================

st.sidebar.subheader(
    "🕘 Recent Conversations"
)


if not threads:

    st.sidebar.write(
        "No past conversations yet."
    )

else:

    for index, past_thread_id in enumerate(
        threads
    ):

        past_thread_id = str(
            past_thread_id
        )

        # Highlight current thread
        if past_thread_id == thread_key:

            label = (
                f"🟢 CURRENT  ·  "
                f"{past_thread_id[:8]}..."
            )

        else:

            label = (
                f"🗨️  {past_thread_id[:8]}..."
            )

        if st.sidebar.button(
            label,
            key=f"side-thread-{index}-{past_thread_id}",
            use_container_width=True
        ):

            if past_thread_id != thread_key:

                st.session_state["thread_id"] = (
                    past_thread_id
                )

                st.session_state[
                    "message_history"
                ] = load_conversation(
                    past_thread_id
                )

                st.session_state[
                    "pending_approval"
                ] = None

                st.rerun()


# ============================================================
# MAIN CHAT
# ============================================================

st.title(
    "💬 DroX Chatbot"
)


# ------------------------------------------------------------
# SHOW CURRENT THREAD IN MAIN CHAT
# ------------------------------------------------------------

st.caption(
    f"🟢 **Current Thread:** `{thread_key}`"
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state[
    "message_history"
]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# HITL APPROVAL UI
# ============================================================

pending_approval = (
    st.session_state["pending_approval"]
)


if pending_approval:

    st.warning(
        "⚠️ Human approval required"
    )

    question = pending_approval.get(
        "question",
        "Please approve or deny this action."
    )

    symbol = pending_approval.get(
        "symbol",
        ""
    )

    quantity = pending_approval.get(
        "quantity",
        ""
    )

    st.markdown(
        f"### {question}"
    )

    if symbol or quantity:

        st.info(
            f"Stock: **{symbol}**  \n"
            f"Quantity: **{quantity} shares**"
        )

    col1, col2 = st.columns(2)


    # ========================================================
    # APPROVE
    # ========================================================

    with col1:

        if st.button(
            "✅ Approve Purchase",
            use_container_width=True,
            type="primary"
        ):

            with st.spinner(
                "Processing approved purchase..."
            ):

                result = resume_chat(
                    thread_id=thread_key,
                    decision="yes"
                )

            if result.get("status") == "interrupt":

                st.session_state[
                    "pending_approval"
                ] = result.get(
                    "approval"
                )

                st.rerun()

            elif result.get("status") == "completed":

                final_message = result.get(
                    "message",
                    ""
                )

                if final_message:

                    st.session_state[
                        "message_history"
                    ].append(
                        {
                            "role": "assistant",
                            "content": final_message
                        }
                    )

                st.session_state[
                    "pending_approval"
                ] = None

                st.rerun()

            else:

                st.error(
                    result.get(
                        "message",
                        "Something went wrong."
                    )
                )


    # ========================================================
    # DENY
    # ========================================================

    with col2:

        if st.button(
            "❌ Deny Purchase",
            use_container_width=True
        ):

            with st.spinner(
                "Cancelling purchase..."
            ):

                result = resume_chat(
                    thread_id=thread_key,
                    decision="no"
                )

            if result.get("status") == "interrupt":

                st.session_state[
                    "pending_approval"
                ] = result.get(
                    "approval"
                )

                st.rerun()

            elif result.get("status") == "completed":

                final_message = result.get(
                    "message",
                    ""
                )

                if final_message:

                    st.session_state[
                        "message_history"
                    ].append(
                        {
                            "role": "assistant",
                            "content": final_message
                        }
                    )

                st.session_state[
                    "pending_approval"
                ] = None

                st.rerun()

            else:

                st.error(
                    result.get(
                        "message",
                        "Something went wrong."
                    )
                )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state["pending_approval"]:

    st.info(
        "Please approve or deny the pending purchase "
        "before sending another message."
    )

    user_input = None

else:

    user_input = st.chat_input(
        "Ask about your documents or use tools..."
    )


# ============================================================
# HANDLE USER INPUT
# ============================================================

if user_input:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(
            user_input
        )


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(
            "### 🔎 Agent Activity"
        )

        activity_placeholder = st.empty()

        result = run_chat_with_live_updates(
            user_input=user_input,
            thread_id=thread_key,
            activity_placeholder=activity_placeholder
        )


    # ========================================================
    # HITL INTERRUPT
    # ========================================================

    if result.get("status") == "interrupt":

        st.session_state[
            "pending_approval"
        ] = result.get(
            "approval"
        )

        st.rerun()


    # ========================================================
    # NORMAL RESPONSE
    # ========================================================

    elif result.get("status") == "completed":

        ai_message = result.get(
            "message",
            ""
        )

        if ai_message:

            st.session_state[
                "message_history"
            ].append(
                {
                    "role": "assistant",
                    "content": ai_message
                }
            )

            st.rerun()


    # ========================================================
    # ERROR
    # ========================================================

    else:

        error_message = result.get(
            "message",
            "Something went wrong."
        )

        with st.chat_message(
            "assistant"
        ):

            st.error(
                error_message
            )


# ============================================================
# DOCUMENT METADATA
# ============================================================

doc_meta = thread_document_metadata(
    thread_key
)

if doc_meta:

    st.caption(
        f"Document indexed: "
        f"{doc_meta.get('filename')} "
        f"(chunks: {doc_meta.get('chunks')}, "
        f"pages: {doc_meta.get('documents')})"
    )


# ============================================================
# DIVIDER
# ============================================================

st.divider()