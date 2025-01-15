import streamlit as st
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# Page configuration
st.set_page_config(page_title="YouTube Video Chat", layout="wide")
st.title("💬 YouTube Video Chat Assistant")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chain' not in st.session_state:
    st.session_state.chain = None

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    query = parse_qs(urlparse(url).query)
    return query.get('v', [None])[0]

def load_video_transcript(video_url):
    """Load and process YouTube video transcript"""
    try:
        video_id = get_video_id(video_url)
        if not video_id:
            st.error("❌ Could not extract video ID from URL")
            return None
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = ' '.join([entry['text'] for entry in transcript_list])
        
        return Document(
            page_content=full_text,
            metadata={"source": video_url, "video_id": video_id}
        )
    
    except Exception as e:
        st.error(f"❌ Error loading transcript: {str(e)}")
        return None

def setup_qa_chain(document):
    """Set up the QA chain with the document"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents([document])

        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        vector_store = FAISS.from_documents(texts, embeddings)
        st.session_state.vector_store = vector_store

        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-002",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.7,
            top_k=3,
            top_p=0.8
        )

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )
        
        st.session_state.chain = chain
        return True

    except Exception as e:
        st.error(f"❌ Error setting up QA chain: {str(e)}")
        return False

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    if st.button("Load Video", key="load_button"):
        with st.spinner("Loading transcript..."):
            document = load_video_transcript(video_url)
            if document:
                with st.spinner("Setting up QA system..."):
                    if setup_qa_chain(document):
                        st.success("✅ Ready to chat!")
                        # Show transcript preview
                        with st.expander("📝 Transcript Preview"):
                            st.text(document.page_content[:500] + "...")
    
    if st.button("Clear Chat", key="clear_button"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if st.session_state.chain:
    st.header("🤖 Chat with the Video Content")
    
    # Chat input
    user_question = st.text_input("Ask a question about the video:", key="user_input")
    
    if st.button("Send", key="send"):
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    # Get response from chain
                    response = st.session_state.chain({"question": user_question})
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response["answer"]
                    })
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💭 Chat History")
        for i, exchange in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {exchange['question']}")
                st.markdown(f"**Assistant:** {exchange['answer']}")
                st.divider()
else:
    st.info("👋 Enter a YouTube URL in the sidebar to start chatting about the video!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and LangChain")
