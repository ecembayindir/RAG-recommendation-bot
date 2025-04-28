import streamlit as st
import pandas as pd
from genai_project import BookRecommenderBot, BOT_ROLES

def main():
    st.set_page_config(page_title="Book Recommendation Chatbot", page_icon="📚", layout="wide")
    st.title("📚 Book Recommendation Chatbot")

    if 'bot' not in st.session_state:
        st.session_state.bot = BookRecommenderBot()
        test_response = st.session_state.bot.llm("Test")
        if "Sorry, no AI model is available" in test_response:
            st.error("AI model unavailable. Please ensure LM Studio is running on localhost:1234 or check your internet connection for HuggingFace access. Responses will be limited to retrieved data.")

    with st.sidebar:
        st.header("Bot Configuration")
        st.subheader("Select Bot Role")
        role_options = {role_info["name"]: role_key for role_key, role_info in BOT_ROLES.items()}
        selected_role_name = st.selectbox(
            "Choose a recommendation style:",
            options=list(role_options.keys()),
            index=1
        )
        selected_role = role_options[selected_role_name]

        if 'current_role' not in st.session_state or st.session_state.current_role != selected_role:
            st.session_state.current_role = selected_role
            st.session_state.bot.set_role(selected_role)

        st.markdown(f"**Current Role:** {st.session_state.bot.get_role_description()}")

        st.header("Book Filters")
        df_all = pd.read_parquet("data/books_dataset.parquet")
        genres = sorted(set(g.strip() for genre_list in df_all["genre"].dropna() for g in genre_list.split(",")))
        authors = df_all["author"].dropna().unique().tolist()[:100]

        selected_genres = st.multiselect("Select Genres", genres)
        selected_authors = st.multiselect("Select Authors", authors)
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)

        if st.button("Apply Filters"):
            st.session_state.df = st.session_state.bot.get_filtered_dataframe(
                genres=selected_genres,
                authors=selected_authors,
                min_rating=min_rating
            )
            st.success(f"Filtered to {len(st.session_state.df)} books")

        st.header("Upload Additional Documents")
        uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
        if uploaded_file:
            with open("temp_file", "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = st.session_state.bot.load_external_document("temp_file")
            st.success(result)

    if 'df' not in st.session_state:
        st.session_state.df = st.session_state.bot.df

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Conversation", "Book Details", "Chat History", "Help", "How It Works"])

    with tab1:
        st.markdown(f"**You're talking to: {BOT_ROLES[st.session_state.bot.role]['name']}**")
        user_question = st.text_input(
            "Ask about books (e.g., 'Books about revolution' or 'Recommend a sci-fi book')",
            key="user_question"
        )

        if st.button("Clear Chat History"):
            st.session_state.bot.clear_history()
            st.experimental_rerun()

        if user_question:
            with st.spinner("Thinking..."):
                df_to_use = st.session_state.df if 'df' in st.session_state else None
                answer, sources = st.session_state.bot.ask_question(user_question, df_to_use)

            st.markdown("### Answer:")
            if "Sorry, no AI model is available" in answer:
                st.warning(answer)
            else:
                st.markdown(answer)

            if sources:
                st.markdown("### Sources:")
                for i, doc in enumerate(sources):
                    st.write(f"{i+1}. **{doc.metadata['title']}** by {doc.metadata['author']} (Rating: {doc.metadata['rating']})")

    with tab2:
        if hasattr(st.session_state.bot, 'db_response') and st.session_state.bot.db_response:
            st.markdown("### Book Details Used for Recommendation:")
            for i, doc in enumerate(st.session_state.bot.db_response):
                with st.expander(
                        f"📖 {doc.metadata['title']} by {doc.metadata.get('author', 'Unknown')} | Rating: {doc.metadata.get('rating', 'N/A')}"):
                    st.markdown("**Description:**")
                    desc = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
                    st.markdown(desc)
                    st.markdown(f"**Genre:** {doc.metadata.get('genre', 'Unknown')}")
                    st.markdown(f"**Pages:** {doc.metadata.get('pages', 'Unknown')}")
        else:
            st.info("Ask a question to see book details.")

    with tab3:
        st.markdown("### Chat History:")
        if not st.session_state.bot.chat_history:
            st.info("No chat history yet.")
        else:
            for i, (question, answer) in enumerate(st.session_state.bot.chat_history):
                st.markdown(f"**User:**")
                st.markdown(f"{question}")
                st.markdown(f"**Bot:**")
                st.markdown(f"{answer}")
                st.markdown("---")

    with tab4:
        st.markdown("### Help & Instructions")
        st.markdown("""
        This book recommender bot can act in different roles to provide personalized recommendations:

        #### Available Roles:
        - **Literary Critic**: For analytical, in-depth book discussions
        - **Casual Reader**: For easy, entertaining reads
        - **Academic**: For scholarly works and intellectual content
        - **Genre Specialist**: For genre-specific recommendations

        #### Commands:
        - Type `/role role_name` to change the bot's role (e.g., `/role academic`)
        - Select a role from the sidebar

        #### Features:
        - Filter by genre, author, and rating
        - Upload PDFs or TXT files to augment recommendations
        - View chat history and sources

        #### Example Questions:
        - "Books about revolution"
        - "Summarize Hungary 56"
        - "Recommend history books"
        """)

    with tab5:
        st.markdown("### How This Generative AI Works")
        st.markdown("""
        This chatbot uses **Retrieval-Augmented Generation (RAG)**:

        #### 1. Role-Based Personas
        - Four roles with unique tones: **Literary Critic**, **Casual Reader**, **Academic**, **Genre Specialist**.

        #### 2. RAG Workflow
        - Loads `books_dataset.parquet`, splits descriptions into chunks (2000 chars), embeds with `all-MiniLM-L6-v2`, stores in FAISS.
        - Retrieves top 5 relevant chunks, generates answers with DeepSeek (or fallback).

        #### 3. Performance
        - Index cached at startup or filter change for fast queries.
        - LLM runs locally via LM Studio.
        """)


if __name__ == "__main__":
    main()