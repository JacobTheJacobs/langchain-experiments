import pytube
import requests
import re
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import tiktoken
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from langchain.schema import Document
import yt_dlp  # For searching and downloading YouTube videos
import time  # For sleep delays to avoid rate limiting
from transformers import pipeline  # For extracting key points
import chromadb
from langchain_community.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer


# Initialize Chroma with the new settings
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="youtube_videos")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another model if preferred


# Function to search YouTube for videos related to a specific topic
def search_youtube_videos(query: str, max_results: int = 5):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'force_generic_extractor': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        return [{'url': entry['webpage_url'], 'title': entry['title']} for entry in search_results['entries']]

def get_youtube_description(url: str):
    full_html = requests.get(url).text
    y = re.search(r'shortDescription":"', full_html)
    desc = ""
    count = y.start() + 19  # adding the length of the 'shortDescription":"
    while True:
        letter = full_html[count]
        if letter == "\"":
            if full_html[count - 1] == "\\":
                desc += letter
                count += 1
            else:
                break
        else:
            desc += letter
            count += 1
    return desc

def get_youtube_info(url: str):
    yt = pytube.YouTube(url)
    title = yt.title
    if title is None:
        title = "None"
    desc = get_youtube_description(url)
    if desc is None:
        desc = "None"
    return title, desc

def get_youtube_transcript_loader_langchain(url: str, language='en'):
    video_id = re.search(r'v=([a-zA-Z0-9_-]{11})', url).group(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
    except NoTranscriptFound:
        available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = available_transcripts.find_generated_transcript(['iw']).fetch()  # Fetching Hebrew (auto-generated) transcript
    transcript_text = " ".join([t['text'] for t in transcript])
    return transcript_text

def wrap_docs_to_string(docs):
    return " ".join([doc.page_content for doc in docs]).strip()

def get_text_splitter(chunk_size: int, overlap_size: int):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=overlap_size)

def get_youtube_transcription(url: str):
    text = get_youtube_transcript_loader_langchain(url)
    enc = tiktoken.encoding_for_model("gpt-4")
    count = len(enc.encode(text))
    return text, count

def clean_transcription(text):
    # Remove unnecessary characters or noise from the transcription
    cleaned_text = re.sub(r'\[.*?\]', '', text)  # Remove anything in brackets
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
    return cleaned_text.strip()

def get_transcription_summary(url: str, temperature: float, chunk_size: int, overlap_size: int):
    transcript_text = get_youtube_transcript_loader_langchain(url)
    transcript_text = clean_transcription(transcript_text)
    docs = [Document(page_content=transcript_text)]
    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    split_docs = text_splitter.split_documents(docs)
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=temperature,
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    output = chain.invoke(split_docs)
    return output['output_text']

def store_to_chroma(video_id, title, url, transcript, summary):
    # Generate embedding for the summary using SentenceTransformer
    summary_embedding = embedding_model.encode(summary)
    
    # Convert the NumPy array to a list
    summary_embedding = summary_embedding.tolist()

    # Store the video information in Chroma
    collection.add(
        documents=[summary],
        embeddings=[summary_embedding],
        metadatas=[{
            'video_id': video_id,
            'title': title,
            'url': url,
            'transcript': transcript
        }],
        ids=[video_id]
    )


def extract_key_points(summary_text):
    key_point_extractor = pipeline("text-classification", model="facebook/bart-large-mnli")
    key_points = key_point_extractor(summary_text)
    return key_points

def process_video(url, temperature, chunk_size, overlap_size):
    # Step 1: Download video info and extract transcript
    title, desc = get_youtube_info(url)
    print(f"Processing Video: {title}\nDescription: {desc}\n")

    transcript_text, token_count = get_youtube_transcription(url)
    print(f"Transcription (Token Count: {token_count}):\n{transcript_text}\n")

    # Step 2: Summarize the transcript
    summary = get_transcription_summary(url, temperature, chunk_size, overlap_size)
    print(f"Summary:\n{summary}\n")

    # Step 3: Store data in Chroma
    video_id = re.search(r'v=([a-zA-Z0-9_-]{11})', url).group(1)
    store_to_chroma(video_id, title, url, transcript_text, summary)

def generate_final_report(output_file="final_report.txt"):
    report = ""
    all_documents = collection.get()['documents']
    all_metadata = collection.get()['metadatas']

    for doc, metadata in zip(all_documents, all_metadata):
        key_points = extract_key_points(doc)
        report += f"Title: {metadata['title']}\n"
        report += f"URL: {metadata['url']}\n"
        report += f"Summary: {doc}\n"
        report += f"Key Points: {key_points}\n"
        report += "\n\n"

    # Write the report to a text file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(report)
    
    print(f"Final report written to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize YouTube videos related to a specific topic")
    parser.add_argument("topic", type=str, help="Topic to search for on YouTube")
    parser.add_argument("--max_results", type=int, default=5, help="Maximum number of videos to process")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for LLM")
    parser.add_argument("--chunk_size", type=int, default=4000, help="Chunk size for text splitter")
    parser.add_argument("--overlap_size", type=int, default=0, help="Overlap size for text splitter")
    parser.add_argument("--output_file", type=str, default="final_report.txt", help="Output file for the final report")

    args = parser.parse_args()

    topic = args.topic
    max_results = args.max_results
    temperature = args.temperature
    chunk_size = args.chunk_size
    overlap_size = args.overlap_size
    output_file = args.output_file

    # Search YouTube for videos on the topic
    videos = search_youtube_videos(topic, max_results=max_results)
    for video in videos:
        url = video['url']
        process_video(url, temperature, chunk_size, overlap_size)
        time.sleep(2)  # Adding a delay to avoid rate limiting

    # Generate and print the final report
    generate_final_report(output_file)

#   init research
#   python main_search.py "Artificial Intelligence" --max_results 3 --temperature 0.5 --chunk_size 2000 --overlap_size 200 --output_file ai_report.txt
