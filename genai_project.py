import pandas as pd
import os
import time
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize embedding model
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define bot personas/roles
BOT_ROLES = {
    "literary_critic": {
        "name": "Literary Critic",
        "description": "I provide in-depth analysis of books, focusing on themes, style, and cultural impact.",
        "prompt_template": "As a literary critic with a formal tone, use the following context to answer. If '{question}' is a book title, provide a detailed summary or analysis; otherwise, evaluate critically and identify books related to: {question}. Provide only the final response without internal reasoning or <think> tags.\nContext: {context}",
        "tone": "formal",
        "max_tokens": 700
    },
    "casual_reader": {
        "name": "Casual Reader",
        "description": "I suggest fun, easy reads for relaxation.",
        "prompt_template": "Hey there! As a casual reader with a friendly tone, use the following context to answer. If '{question}' is a book title, give a quick summary or suggest similar fun reads; otherwise, recommend cool books related to: {question}. Provide only the final response without internal reasoning or <think> tags.\nContext: {context}",
        "tone": "friendly",
        "max_tokens": 300
    },
    "academic": {
        "name": "Academic",
        "description": "I recommend scholarly works and dive into intellectual discussions.",
        "prompt_template": "As an academic expert with a scholarly tone, use the following context to answer. If '{question}' is a book title, provide a scholarly summary or key insights; otherwise, address with rigor and suggest books related to: {question}. Provide only the final response without internal reasoning or <think> tags.\nContext: {context}",
        "tone": "scholarly",
        "max_tokens": 600
    },
    "genre_specialist": {
        "name": "Genre Specialist",
        "description": "I specialize in genre-specific recommendations with deep expertise.",
        "prompt_template": "As a genre specialist with an enthusiastic tone, use the following context to answer. If '{question}' is a book title, offer a genre-specific summary or recommendations; otherwise, respond with genre expertise and recommend books related to: {question}. Provide only the final response without internal reasoning or <think> tags.\nContext: {context}",
        "tone": "enthusiastic",
        "max_tokens": 500
    }
}


class BookRecommenderBot:
    def __init__(self, role="casual_reader"):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        self.role = role
        self.db = None
        self.llm = self.load_llm()
        self.qa_chain = None
        self.df = self.get_filtered_dataframe()
        self._initialize_qa_chain()

    def load_llm(self):
        """Load the local DeepSeek LLM via LM Studio"""
        print("Attempting to connect to DeepSeek via LM Studio on localhost:1234...")
        try:
            llm = ChatOpenAI(
                openai_api_base="http://localhost:1234/v1",
                openai_api_key="not-needed",
                model="deepseek-r1-distill-qwen-7b",
                temperature=0.7,
                max_tokens=512,
                timeout=300
            )
            response = llm.invoke("Hello")
            print(f"Success: Connected to DeepSeek on localhost:1234. Test response: {response.content}")
            return llm
        except Exception as e:
            print(f"Failed to connect to DeepSeek on localhost:1234. Error: {str(e)}")
            print("Falling back to HuggingFace...")
            try:
                from langchain.llms import HuggingFaceHub
                token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not token:
                    print("Warning: HUGGINGFACEHUB_API_TOKEN not found in .env.")
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    huggingfacehub_api_token=token,
                    model_kwargs={"max_length": 512}
                )
                response = llm("Hello")
                print(f"Success: Connected to HuggingFace (google/flan-t5-large). Test response: {response}")
                return llm
            except Exception as hf_e:
                print(f"Failed to connect to HuggingFace. Error: {hf_e}")

                class DummyLLM:
                    def __call__(self, prompt, retrieved_docs=None):
                        base_response = "Sorry, no AI model is available. Ensure LM Studio is running on localhost:1234 or check your HuggingFace token."
                        if retrieved_docs:
                            doc_summary = "\nHere’s what I found in the data:\n" + "\n".join(
                                [f"- {doc.metadata['title']} by {doc.metadata['author']}: {doc.page_content[:100]}..."
                                 for doc in retrieved_docs]
                            )
                            return base_response + doc_summary
                        return base_response

                print("Using Dummy LLM as final fallback.")
                return DummyLLM()

    def set_role(self, role):
        if role in BOT_ROLES:
            self.role = role
            return f"Role changed to: {BOT_ROLES[role]['name']}"
        return f"Invalid role. Available roles: {', '.join(BOT_ROLES.keys())}"

    def get_filtered_dataframe(self, genres=None, authors=None, min_rating=0):
        start_time = time.time()
        df = pd.read_parquet("data/books_dataset.parquet")
        df = df.dropna(subset=["title", "desc", "author", "genre", "rating", "pages"])
        df = df.drop_duplicates(subset=["title", "author"], keep="first")
        if genres:
            df = df[df["genre"].apply(lambda x: any(g in x.split(",") for g in genres))]
        if authors:
            df = df[df["author"].isin(authors)]
        df = df[df["rating"] >= min_rating]
        df = df.reset_index(drop=True)
        print(f"Dataframe filtered in {time.time() - start_time:.2f} seconds")
        return df.head(500)

    def _initialize_qa_chain(self):
        start_time = time.time()
        documents = [
            Document(page_content=row["desc"],
                     metadata={"title": row["title"],
                               "author": row["author"],
                               "pages": row["pages"],
                               "rating": row["rating"],
                               "genre": row["genre"]})
            for _, row in self.df.iterrows()
        ]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(split_docs, embeddings)

        if self.db:
            db.merge_from(self.db)
        self.db = db

        retriever = self.db.as_retriever(search_kwargs={"k": 5})

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            return_generated_question=True,
            get_chat_history=lambda h: h
        )
        print(f"QA chain initialized in {time.time() - start_time:.2f} seconds")

    def load_external_document(self, file_path):
        start_time = time.time()
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            return "Unsupported file format. Use PDF or TXT."

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        if not self.db:
            self.db = FAISS.from_documents(split_docs, embeddings)
        else:
            self.db.merge_from(FAISS.from_documents(split_docs, embeddings))
        self._initialize_qa_chain()
        print(f"External document loaded in {time.time() - start_time:.2f} seconds")
        return f"Loaded {len(split_docs)} document chunks from {file_path}"

    def ask_question(self, question, df=None):
        start_time = time.time()
        if not question:
            return "Please ask a question about books.", []
        if len(question.split()) < 2 and not question.startswith("/role "):
            return "Please provide more detail in your question (e.g., 'Books about revolution' or 'Recommend a sci-fi book').", []

        if question.startswith("/role "):
            new_role = question.replace("/role ", "").strip()
            response = self.set_role(new_role)
            return response, []

        if df is not None and not df.equals(self.df):
            self.df = df
            self._initialize_qa_chain()

        role_info = BOT_ROLES.get(self.role, BOT_ROLES["casual_reader"])

        # Retrieve relevant documents
        retrieve_start = time.time()
        retrieved_docs = self.qa_chain.retriever.get_relevant_documents(question)
        unique_docs = {}
        for doc in retrieved_docs:
            title = doc.metadata["title"]
            if title not in unique_docs:
                unique_docs[title] = doc
        deduped_docs = list(unique_docs.values())
        context = "\n".join([f"{doc.metadata['title']}: {doc.page_content}" for doc in deduped_docs])
        print(f"Retrieval took {time.time() - retrieve_start:.2f} seconds")

        # Prepare the prompt for the LLM
        role_prompt = role_info["prompt_template"].format(question=question, context=context)

        # Generate the response
        try:
            llm_start = time.time()
            result = self.qa_chain({"question": role_prompt, "chat_history": self.chat_history})
            print(f"LLM inference took {time.time() - llm_start:.2f} seconds")
            raw_answer = result["answer"]
            generated_question = result.get("generated_question", role_prompt)
        except Exception as e:
            print(f"Error in QA chain: {e}")
            if isinstance(self.llm, type(lambda x: x)) and hasattr(self.llm, '__call__'):
                raw_answer = self.llm(role_prompt, retrieved_docs=deduped_docs)
            else:
                raw_answer = f"Sorry, I couldn’t process your request due to a technical issue: {str(e)}."
            # Define a fallback for result in case of exception
            result = {"answer": raw_answer}  # Minimal dict to avoid UnboundLocalError
            generated_question = role_prompt  # Fallback to the prompt itself

        # Clean the response by removing <think> and </think> blocks
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()

        # Add role and tone metadata
        enhanced_answer = f"{clean_answer}\n\n[Responding as: {role_info['name']} | Tone: {role_info['tone']}]"
        if not deduped_docs:
            enhanced_answer += "\n\n(Note: Limited information found; response may be incomplete.)"

        # Update internal state
        self.chat_history.append((question, enhanced_answer))
        self.db_query = generated_question  # Use the safely assigned variable
        self.db_response = deduped_docs
        self.answer = enhanced_answer

        print(f"Total query time: {time.time() - start_time:.2f} seconds")
        print("Raw LLM output:", raw_answer)
        print("Retrieved documents:", [doc.metadata["title"] for doc in deduped_docs])
        return self.answer, self.db_response

    def clear_history(self):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []

    def get_role_description(self):
        return BOT_ROLES[self.role]["description"]


if __name__ == "__main__":
    bot = BookRecommenderBot(role="casual_reader")
    df = bot.get_filtered_dataframe()
    answer, sources = bot.ask_question("Books including revolution", df)
    print("Answer:", answer)
    print("Sources:", [doc.metadata["title"] for doc in sources])