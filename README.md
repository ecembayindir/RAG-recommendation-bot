<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <h1 align="center">📚 RAG Book Recommendation Bot with DeepSeek LLM</h1>

<h2>🛠 Technology Stack</h2>
<div class="badges">
  <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/LangChain-FF6F61?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/FAISS-4B8BBE?style=for-the-badge&logo=faiss&logoColor=white" alt="FAISS"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Sentence_Transformers-6B7280?style=for-the-badge&logo=huggingface&logoColor=white" alt="Sentence Transformers"/>
  <img src="https://img.shields.io/badge/HuggingFace-F4A261?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/DeepSeek-0078D4?style=for-the-badge&logo=ai&logoColor=white" alt="DeepSeek"/>
</div>

  <h2>🚀 <strong>Objective</strong></h2>
  <p>
    This project develops a Retrieval-Augmented Generation (RAG) book recommendation chatbot powered by a local DeepSeek generative AI model via LM Studio. It offers personalized book recommendations through four distinct personas—Literary Critic, Casual Reader, Academic, and Genre Specialist—using a curated dataset of books and optional external documents (PDF/TXT). The chatbot is accessible via an interactive Streamlit web interface, with features like genre/author filtering and conversation history.
  </p>

  <h2>📂 <strong>Project Summary</strong></h2>
  <p>
    The Book Recommendation Chatbot uses a RAG pipeline to retrieve relevant book data from a FAISS vector store and generate responses with DeepSeek. It offers:
  </p>
  <ul>
    <li><strong>Four distinct personas</strong> with unique tones and recommendation styles</li>
    <li><strong>Filtering options</strong> by genre, author, and rating</li>
    <li><strong>Document upload capability</strong> to enhance recommendations with custom content</li>
    <li><strong>User-friendly Streamlit interface</strong> with dedicated tabs for conversation, book details, chat history, and help</li>
  </ul>

  <h2>🛠️ <strong>Technology Stack</strong></h2>
  <table>
    <thead>
      <tr>
        <th>Technology</th>
        <th>Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Python 3.9+</strong></td>
        <td>Core programming language</td>
      </tr>
      <tr>
        <td><strong>Streamlit</strong></td>
        <td>Interactive web interface</td>
      </tr>
      <tr>
        <td><strong>LangChain</strong></td>
        <td>RAG pipeline orchestration</td>
      </tr>
      <tr>
        <td><strong>FAISS</strong></td>
        <td>Vector database for similarity search</td>
      </tr>
      <tr>
        <td><strong>Sentence Transformers</strong></td>
        <td><code>all-MiniLM-L6-v2</code> for text embeddings</td>
      </tr>
      <tr>
        <td><strong>DeepSeek</strong></td>
        <td><code>deepseek-r1-distill-qwen-7b</code> model for text generation</td>
      </tr>
      <tr>
        <td><strong>HuggingFace</strong></td>
        <td>Fallback model API integration</td>
      </tr>
      <tr>
        <td><strong>Pandas</strong></td>
        <td>Data manipulation and preprocessing</td>
      </tr>
    </tbody>
  </table>

  <h2>📸 <strong>Screenshots</strong></h2>
  <p>
    Below are example screenshots showcasing the RAG process and responses from different bot personas.
  </p>

  <p><strong>RAG Process</strong>: Illustrates the retrieval and generation workflow.<br>
    <img src="https://i.imgur.com/r1KZEDC.png" alt="RAG Process" width="600"/>
  </p>

  <p><strong>Literary Critic Response</strong>: Example response with a formal, analytical tone.<br>
    <img src="https://i.imgur.com/E8PUT9P.png" alt="Literary Critic Response" width="600"/>
  </p>

  <p><strong>Casual Reader Response</strong>: Example response with a friendly, approachable tone.<br>
    <img src="https://i.imgur.com/kR2b3KX.png" alt="Casual Reader Response" width="600"/>
  </p>

  <h2>📈 <strong>Methodology</strong></h2>

  <h3>Data Processing Pipeline</h3>
  <ol>
    <li><strong>Data Loading</strong>: Loads <code>books_dataset.parquet</code> and pre-processes metadata</li>
    <li><strong>Text Chunking</strong>: Splits book descriptions into 2000-character chunks with 200-character overlap</li>
    <li><strong>Embedding</strong>: Transforms chunks into vector representations using <code>all-MiniLM-L6-v2</code></li>
    <li><strong>Indexing</strong>: Stores embeddings in a FAISS vector database for efficient similarity search</li>
    <li><strong>Retrieval</strong>: Fetches top 5 most relevant text chunks for user queries</li>
    <li><strong>Generation</strong>: Produces contextual responses using DeepSeek or fallback LLM</li>
  </ol>

  <h3>Model Architecture</h3>
  <ul>
    <li>
      <strong>Primary LLM</strong>: DeepSeek model (<code>deepseek-r1-distill-qwen-7b</code>)
      <ul>
        <li>Served locally via LM Studio</li>
        <li>Configuration: temperature 0.7, max tokens 512</li>
      </ul>
    </li>
    <li>
      <strong>Fallback Models</strong>:
      <ul>
        <li>HuggingFace <code>google/flan-t5-large</code> (requires API token)</li>
        <li>Simple extractive fallback that summarizes retrieved documents</li>
      </ul>
    </li>
    <li>
      <strong>Embedding Model</strong>:
      <ul>
        <li><code>sentence-transformers/all-MiniLM-L6-v2</code> for semantic text representation</li>
      </ul>
    </li>
  </ul>

  <h2>🔍 <strong>Implementation Details</strong></h2>

  <h3>Persona System</h3>
  <table>
    <thead>
      <tr>
        <th>Persona</th>
        <th>Tone</th>
        <th>Focus</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Literary Critic</td>
        <td>Formal, analytical</td>
        <td>Literary merit, themes, context</td>
      </tr>
      <tr>
        <td>Casual Reader</td>
        <td>Friendly, conversational</td>
        <td>Enjoyment, readability, emotional impact</td>
      </tr>
      <tr>
        <td>Academic</td>
        <td>Scholarly, thorough</td>
        <td>Research value, historical context</td>
      </tr>
      <tr>
        <td>Genre Specialist</td>
        <td>Enthusiastic, knowledgeable</td>
        <td>Genre conventions, comparisons</td>
      </tr>
    </tbody>
  </table>

  <h3>RAG Implementation</h3>
  <pre><code>User Query → Embedding → Vector Search → Context Retrieval → Prompt Construction → LLM Generation</code></pre>

  <p>Each response is generated by:</p>
  <ol>
    <li>Converting the user's query to an embedding vector</li>
    <li>Finding semantically similar book chunks from the FAISS index</li>
    <li>Constructing a persona-specific prompt with retrieved context</li>
    <li>Generating a tailored response with DeepSeek or fallback model</li>
  </ol>

  <h3>Deployment Architecture</h3>
  <ul>
    <li><strong>Local Server</strong>: Streamlit application runs on <code>localhost:8501</code></li>
    <li><strong>LLM Server</strong>: DeepSeek model must be running via LM Studio on <code>localhost:1234</code></li>
    <li>
      <strong>Configuration</strong>: Environment variables stored in <code>.env</code> file
      <ul>
        <li><code>HUGGINGFACEHUB_API_TOKEN</code>: Required for HuggingFace fallback LLM</li>
      </ul>
    </li>
  </ul>

  <h2>📊 <strong>Key Features</strong></h2>
  <ul>
    <li><strong>Role-Based Personas</strong>: Four distinct recommendation styles</li>
    <li><strong>Semantic Search</strong>: Finds relevant books based on meaning, not just keywords</li>
    <li><strong>Multi-criteria Filtering</strong>: Filter by genre, author, and minimum rating</li>
    <li><strong>Knowledge Augmentation</strong>: Upload PDF/TXT files to enhance recommendations</li>
    <li><strong>Interactive UI</strong>: Tabbed interface with conversation, book details, and history views</li>
    <li><strong>Fallback System</strong>: Graceful degradation when DeepSeek is unavailable</li>
  </ul>

  <h2>📚 <strong>Dataset Structure</strong></h2>
  <p>The project uses a curated books dataset (<code>books_dataset.parquet</code>) with the following schema:</p>
  <p>Dataset source: <a href="https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k" target="_blank">Goodreads Books 100K on Kaggle</a></p>

  <table>
    <thead>
      <tr>
        <th>Field</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>title</code></td>
        <td>Book title</td>
        <td>string</td>
      </tr>
      <tr>
        <td><code>desc</code></td>
        <td>Book description/summary</td>
        <td>string</td>
      </tr>
      <tr>
        <td><code>author</code></td>
        <td>Book author(s)</td>
        <td>string</td>
      </tr>
      <tr>
        <td><code>genre</code></td>
        <td>Book genre(s)</td>
        <td>string</td>
      </tr>
      <tr>
        <td><code>rating</code></td>
        <td>Average rating (1-5)</td>
        <td>float</td>
      </tr>
      <tr>
        <td><code>pages</code></td>
        <td>Page count</td>
        <td>integer</td>
      </tr>
    </tbody>
  </table>

  <p>The dataset in the app is limited to 500 high-quality book entries for optimal performance.</p>

  <h2>🚀 <strong>Getting Started</strong></h2>

  <h3>Prerequisites</h3>
  <ul>
    <li>Python 3.9+</li>
    <li>LM Studio with DeepSeek model</li>
    <li>HuggingFace API token (for fallback functionality)</li>
  </ul>

  <h3>Installation</h3>
  <pre><code># Clone the repository
git clone https://github.com/yourusername/book-recommendation-chatbot.git
cd book-recommendation-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "HUGGINGFACEHUB_API_TOKEN=your_token_here" > .env

# Run the application
streamlit run streamlit_app.py</code></pre>

  <h3>Setup Instructions</h3>
  <ol>
    <li>Download and install <a href="https://lmstudio.ai/">LM Studio</a></li>
    <li>Load the <code>deepseek-r1-distill-qwen-7b</code> model in LM Studio</li>
    <li>Start the local server on port 1234</li>
    <li>Launch the Streamlit application</li>
    <li>Select a persona and start chatting!</li>
  </ol>

  <h2>📢 <strong>Conclusion</strong></h2>
  <p>
    This project delivers a robust RAG-based book recommendation chatbot with a local DeepSeek LLM and a Streamlit interface. It supports diverse personas, data filtering, and external document integration, making it versatile for book enthusiasts. The HuggingFace fallback ensures functionality without DeepSeek, though a token is required. Future enhancements could include larger datasets and cloud deployment.
  </p>

  <h2>Connect With Me</h2>
  <div align="center">
    <a href="https://www.linkedin.com/in/ecembayindir" target="_blank">
      <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
    </a>
    <a href="mailto:ecmbyndr@gmail.com">
      <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/>
    </a>
  </div>
  <br>
  <p align="center">© 2025 Ecem Bayindir. All rights reserved.</p>
  <hr/>
  <p align="center">
    <img src="https://komarev.com/ghpvc/?username=ecembayindir&repo=book-recommendation-chatbot&label=Repository%20views&color=0e75b6&style=flat" alt="Repository Views">
  </p>
</body>
</html>
