import yt_dlp
import re
from langchain.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def extract_youtube_transcript(video_url):
    ydl_opts = {
        'skip_download': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'ttml',
        'subtitleslangs': ['en'],
        'outtmpl': 'transcript.%(ext)s'
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Read the downloaded transcript file
        with open('transcript.en.ttml', 'r', encoding='utf-8') as file:
            vtt_content = file.read()
            vtt_content = re.sub(r'<[^>]+>', '', vtt_content)   # Remove HTML tags
            return vtt_content
    except Exception as e:
        print(f"Error extracting transcript: {e}")
        return None

def summarize_transcript(transcript):
    if not transcript:
        return "No transcript available to summarize."

    try:
        llm = Ollama(model="llama3")
        doc = Document(page_content=transcript)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run([doc])
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"

def setup_rag(transcript):
    # Split the transcript into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create a vector store
    db = FAISS.from_texts(texts, embeddings)

    # Create a retriever
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # Create an Ollama LLM
    llm = Ollama(model="llama3")

    # Create a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def ask_question(qa_chain, question):
    result = qa_chain({"query": question})
    return result["result"]

# Example usage
video_url = "X"  # Replace with your desired video URL
transcript = extract_youtube_transcript(video_url)

# Save transcript to file
with open('transcript.txt', 'w', encoding='utf-8') as file:
    file.write(transcript)

if transcript:
    print("Transcript extracted. Length:", len(transcript))
    print("\nFirst 500 characters of transcript:")
    print(transcript[:500])

    print("\nGenerating summary...")
    summary = summarize_transcript(transcript)
    print("\nSummary:")
    print(summary)

    print("\nSetting up RAG system...")
    qa_chain = setup_rag(transcript)

    print("\nYou can now ask questions about the transcript.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
        answer = ask_question(qa_chain, question)
        print("\nAnswer:", answer)

else:
    print("Failed to extract transcript. Please check if the video has auto-generated subtitles.")
