import requests
import logging
import os
from dotenv import load_dotenv
from transformers import pipeline, set_seed, AutoTokenizer
import time
import nltk
from nltk.tokenize import sent_tokenize
from datetime import date

# Download punkt if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Configuration ---
load_dotenv()
FETCH_API_URL_TEMPLATE = "https://api.dailynewshighlights.com/country/{}/summary"
COUNTRIES_FILE = "countries.txt"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
MAX_CHARS = 25000
TARGET_SUMMARY_WORDS = (250, 350)
CHUNK_TARGET_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 150
FINAL_SUMMARY_MIN_TOKENS = 180
FINAL_SUMMARY_MAX_TOKENS = 500
OUTPUT_FILE = "india_summary.txt"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("automation.log", encoding='utf-8'),
                              logging.StreamHandler()])

# --- Functions ---
def get_countries(filename):
    """Reads country list from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            countries = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(countries)} countries from {filename}")
        return countries
    except FileNotFoundError:
        logging.error(f"Error: Country file '{filename}' not found.")
        return []

def fetch_news_data(country):
    """Fetches news data for a single country."""
    url = FETCH_API_URL_TEMPLATE.format(country)
    logging.info(f"Fetching news data for {country} from {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        news_text = response.text.strip()
        if not news_text:
            logging.warning(f"No news content found for {country} in response.")
            return None
        logging.info(f"Successfully fetched data for {country} (text length: {len(news_text)}).")
        return news_text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data for {country}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching data for {country}: {e}")
        return None

def chunk_text_by_sentences(text, tokenizer, max_tokens=CHUNK_TARGET_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """Chunks text into segments based on sentences, trying to stay within token limits."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current_token_count + sentence_tokens <= max_tokens:
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
        else:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                overlap_sentences = current_chunk_sentences[max(0, len(current_chunk_sentences) - (overlap // (len(tokenizer.encode(" ")) + 1))):]
                current_chunk_sentences = list(overlap_sentences)
                current_token_count = len(tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False))
                if current_token_count + sentence_tokens <= max_tokens:
                    current_chunk_sentences.append(sentence)
                    current_token_count += sentence_tokens
                elif chunks and sentence not in chunks[-1]:
                    chunks.append(sentence)
                    current_chunk_sentences = []
                    current_token_count = 0
            elif sentence_tokens > max_tokens:
                sub_sentences = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in [tokenizer.encode(sentence, add_special_tokens=False)[i:i + max_tokens] for i in range(0, sentence_tokens, max_tokens)]]
                chunks.extend(sub_sentences)

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks

def summarize_long_text(text_to_summarize, summarizer_pipeline, tokenizer, target_words=TARGET_SUMMARY_WORDS):
    """Handles long text summarization using chunking and a final pass."""
    if not text_to_summarize:
        logging.warning("No text provided for long text summarization.")
        return None

    chunks = chunk_text_by_sentences(text_to_summarize[:MAX_CHARS], tokenizer)
    intermediate_summaries = []

    logging.info(f"Summarizing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            logging.warning(f"Skipping empty chunk {i+1}/{len(chunks)}")
            continue
        try:
            summary_output = summarizer_pipeline(chunk, max_length=150, min_length=30, do_sample=False, truncation=True)[0]['summary_text']
            intermediate_summaries.append(summary_output)
            logging.info(f"  Summarized chunk {i+1}/{len(chunks)}")
        except Exception as e:
            logging.error(f"Error summarizing chunk {i+1}: {e}")

    combined_summary = " ".join(intermediate_summaries)
    if not combined_summary.strip():
        logging.warning("No intermediate summaries to combine.")
        return None

    logging.info("Generating final summary from combined intermediate summaries...")
    try:
        final_summary_output = summarizer_pipeline(combined_summary,
                                                  max_length=FINAL_SUMMARY_MAX_TOKENS,
                                                  min_length=FINAL_SUMMARY_MIN_TOKENS,
                                                  do_sample=False,
                                                  truncation=True)[0]['summary_text']
        word_count = len(final_summary_output.split())
        logging.info(f"Generated final summary (word count: {word_count}).")
        if not (TARGET_SUMMARY_WORDS[0] <= word_count <= TARGET_SUMMARY_WORDS[1]):
            logging.warning(f"Final summary word count ({word_count}) is outside target range {TARGET_SUMMARY_WORDS}.")
        return final_summary_output
    except Exception as e:
        logging.error(f"Error generating final summary: {e}")
        return None

def save_summary_to_file(summary, filename=OUTPUT_FILE):
    """Saves the generated summary to a text file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(summary)
        logging.info(f"Summary saved to '{filename}'")
        return True
    except Exception as e:
        logging.error(f"Error saving summary to file '{filename}': {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting News Automation Process for India ---")
    start_time = time.time()

    country_to_process = "india"
    summary_content = None

    logging.info(f"Loading summarization model and tokenizer: {SUMMARIZER_MODEL}...")
    try:
        summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
        set_seed(42)
        logging.info("Summarization model and tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load summarization model or tokenizer: {e}. Exiting.")
        exit()

    logging.info(f"\n--- Processing {country_to_process.upper()} ---")
    news_content = fetch_news_data(country_to_process)
    if news_content:
        summary_content = summarize_long_text(news_content, summarizer, tokenizer)
        if summary_content:
            heading = country_to_process.upper().replace('-', ' ').title() + " News Summary"
            logging.info(f"Generated summary for {country_to_process}.")
            save_summary_to_file(f"[{heading}]\n{summary_content}")
        else:
            logging.warning(f"Could not generate summary for {country_to_process}.")
    else:
        logging.warning(f"Skipping {country_to_process} due to fetch error or no data.")

    end_time = time.time()
    logging.info(f"--- News Automation Process Finished in {end_time - start_time:.2f} seconds ---")