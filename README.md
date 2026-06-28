# Historical RAG Assistant

An AI-powered conversational assistant designed to simplify and explain historical events and biographies in an engaging, interactive, and easily accessible manner. The project utilizes a Retrieval-Augmented Generation (RAG) architecture and has evolved across three distinct development phases.
# 🚀 Project Evolution (Versions)
1 🔹 Version 1.0: Core Text-Based RAG Pipeline 
The foundational phase establishes the core retrieval and generation mechanics.

* Vector Storage: Historical data is chunked, embedded, and indexed using ChromaDB for efficient similarity search.
   ```python 
   GEMINI_API_KEY =os.getenv("google_API")
    cl = genai.Client(api_key=GEMINI_API_KEY)

    def get_google_embedding(text):
    result = cl.models.embed_content(
        model="models/gemini-embedding-2",
        config=genai.types.EmbedContentConfig(
            output_dimensionality=768
        ),
        contents=text
    )
    return result.embeddings[0].values
  ```
* HYBRID SEARCH TECHNIQUE: The power of BM25 and Semantic search is combined to introduce good results.
  ```python
     def hybridsearch(self,message, docs):
        pure_texts = docs[0] if docs else []
        langchain_docs = [Document(page_content=text) for text in pure_texts]
    
        bm_25 = BM25.from_documents(langchain_docs)
        bm_25.k = 3 
        vector_retriever = StaticVectorRetriever(docs=langchain_docs)
        assambl = ensambl(
        retrievers=[bm_25, vector_retriever],
        weights=[0.4, 0.6]
        )
        final_docs = assambl.invoke(message)
       return [doc.page_content for doc in final_docs]
  ```  
* LLM Generation: User queries trigger a context lookup from the vector database using ChromaDB, which is an advanced Language Model, then processed via the Groq API to generate accurate, context-aware textual answers.
* Observability: observe the performance of the system by using LangSmith.
  ```python
     from langsmith import traceable
      @traceable(name="second_chaining")
    def llm_send(self,message, history):
        q=self.coll.setup()
        request=self.model.google_model_embed(message)
        embedding = request.embeddings[0].values

  ```
* User Interface: A clean, web-based chat interface built with Gradio to allow seamless text interactions.
  
# 🔹 Version 2.0: Audio Output Integration (Text-to-Speech)
This phase introduces multi-modal capabilities by converting textual explanations into natural spoken audio.

* Arabic Voice Synthesis: Integrates edge-TTS to automatically transform the generated text response into high-quality Arabic speech.
* ```python
      import edge_tts as tts
  ```

* Enhanced Accessibility: Users can now listen to historical narratives rather than just read them, making the learning experience more dynamic.

# 🔹 Version 3.0: Full Voice-to-Voice Interface (Speech-to-Text Integration)
The latest version enables a completely fluid, hands-free voice conversation pipeline.

* Voice Input Processing: Captures the user's spoken question and converts it to text instantly using Whisper (via Groq API).

* In-Memory Streaming: Optimizes performance by processing and buffering audio directly within the system memory (RAM) using Python's io.BytesIO. This bypasses slow disk storage operations (Disk I/O) for ultra-fast, real-time responses.
  ```python
   import io buffer=io.BytesIO()
    wav.write(buffer,sr,record)
    buffer.seek(0)
  ```

* End-to-End Pipeline: Voice Input ➡️ Whisper STT ➡️ ChromaDB RAG Retrieval ➡️ Groq LLM Inference ➡️ Edge-TTS Audio Output.

# 🛠️ Tech Stack
* Language: Python

* Vector Database: ChromaDB

* Inference APIs: Groq API (LLM & Whisper-large-v3), Google API (Embedding)

* Voice Synthesis: edge-tts

* Audio Handling: sounddevice, scipy (In-Memory Buffering)

* UI Framework: Gradio
# 🎯 Key Features
* Simplified History: Transforms complex historical texts into easy-to-understand summaries.

* Zero Disk I/O Latency: Audio recording and playback happen entirely in-memory for maximum speed.

* Multimodal Interaction: Supports text-to-text, text-to-voice, and voice-to-voice conversational flows.

* Hallucination Mitigation: Grounded strictly in verified historical data stored within the local vector space.
