import textwrap
import os
import requests
import pymupdf
from tqdm.auto import tqdm
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import numpy as np

PDF_PATH = "cpd61841.pdf"
PDF_URL = "https://files.support.epson.com/docid/cpd6/cpd61841.pdf"


def check_file_exits(file_path: str, allow_download: bool = False, download_url: str = ""):
    if os.path.exists(file_path):
        print(f"File found: {file_path}")
        return True

    if not allow_download:
        raise ValueError(f"File not found and download not allowed: {file_path}")

    if not download_url:
        raise ValueError("Download URL not provided")

    print(f"File doesn't exist, downloading from {download_url}...")
    try:
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()

        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"File successfully downloaded and saved as {file_path}")
        return True
    except requests.RequestException as e:
        print(f"Failed to download the file: {e}")
        return False


class EmbeddingModel:
    def __init__(self, model_name_or_path="all-mpnet-base-v2", device="cpu"):
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name_or_path,
                                                   device=device)  # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
        self.embedding_model.to("cuda")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pages_info = None
        self.pages_and_chunks = []
        self._initialize_nlp()
        self.embeddings = None  # stores all the embeddings
        self.embeddings_tensor = None

    def _initialize_nlp(self):
        self.nlp = English()
        # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
        self.nlp.add_pipe("sentencizer")

    def _text_formatter(self,text: str) -> str:
        """Performs minor formatting on text."""
        cleaned_text = text.replace("\n", " ").strip()

        return cleaned_text

    def _print_wrapped(self, text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)

    def _get_pages_and_info(self, pdf_path: str):
        doc = pymupdf.open(pdf_path)  # open a document
        pages_info = []
        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = self._text_formatter(text)
            pages_info.append({"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42
                               "page_char_count": len(text),
                               "page_word_count": len(text.split(" ")),
                               "page_sentence_count_raw": len(text.split(". ")),
                               "page_token_count": len(text) / 4,
                               # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                               "text": text})
        return pages_info

    def _tokenize_page(self, page):
        """
            tokenize the page returns sentences
        """
        sentences = list(self.nlp(page["text"]).sents)
        sentences = [str(sentence) for sentence in sentences]

        return sentences

    def _chunk_sentences(self,sentences: list, slice_size: int = 10):
        """
            Group the list of sentences into a group of 10
        """
        return [sentences[i:i + slice_size] for i in range(0, len(sentences), slice_size)]

    def _process_chunk(self,chunk: list,page_number):
        chunk_info = {}
        for item in chunk:
            chunk_info["page_number"] = page_number
            joined_sentence_chunk = "".join(item).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                           joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_info["sentence_chunk"] = joined_sentence_chunk

            # chunk stats
            chunk_info["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_info["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_info["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

        return chunk_info

    def parse_pdf(self):
        """
            Processes stats for the pdf
        """
        self.pages_info = self._get_pages_and_info(pdf_path=PDF_PATH)
        for page in self.pages_info:
            sentences = self._tokenize_page(page=page)
            chunked_sentences = self._chunk_sentences(sentences = sentences)
            page["sentences"] = sentences
            page["sentences_count"] = len(sentences)
            page["sentence_chunks"] = chunked_sentences
            page["sentence_chunks_len"] = len(chunked_sentences)

            self.pages_and_chunks.append(self._process_chunk(chunked_sentences,page_number = page["page_number"]))

    def embed(self):
        check_file_exits(file_path=PDF_PATH, allow_download=True, download_url=PDF_URL)
        self.parse_pdf()
        pages_and_chunks_df = pd.DataFrame(self.pages_and_chunks)
        pages_and_chunks_dict = pages_and_chunks_df.to_dict(orient="records")

        sentence_chunks = [item["sentence_chunk"] for item in pages_and_chunks_dict]
        embeddings = self.embedding_model.encode(sentence_chunks, batch_size=32, convert_to_tensor=True)
        self.embeddings_tensor = torch.tensor(np.array(embeddings.tolist()), dtype=torch.float32).to(self.device)

    def _get_similar(self, query, num=5):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        dot_scores = util.dot_score(a=query_embedding, b=self.embeddings_tensor)[0]

        scores, indices = torch.topk(dot_scores, k=num)
        return scores, indices

    def process_query(self, query):
        pages_and_chunks_df = pd.DataFrame(self.pages_and_chunks)
        pages_and_chunks_dict = pages_and_chunks_df.to_dict(orient="records")

        scores, indices = self._get_similar(query=query)
        print(f"Query : `{query}`")
        for score, index in zip(scores, indices):
            print(f"Score: {score:.3f}")
            print("Text: ")
            self._print_wrapped(pages_and_chunks_dict[index]["sentence_chunk"])
            print("\n")


e_model = EmbeddingModel()
e_model.embed()
e_model.process_query("Issue with ink")
