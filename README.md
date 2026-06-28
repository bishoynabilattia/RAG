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
* LLM Generation: User queries trigger a context lookup from the vector database using ChromaDB, which is an advanced Language Model, then processed via the Groq API to generate accurate, context-aware textual answers.
* Observability: observe the performance of system by using LangSmith.  


* User Interface: A clean, web-based chat interface built with Gradio to allow seamless text interactions.
  
# 🔹 Version 2.0: Audio Output Integration (Text-to-Speech)
This phase introduces multi-modal capabilities by converting textual explanations into natural spoken audio.

* Arabic Voice Synthesis: Integrates edge-TTS to automatically transform the generated text response into high-quality Arabic speech.
* ```python
      import edge_tts as tts```

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
