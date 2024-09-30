from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import  HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "tiiuae/falcon-7b-instruct"

def create_embedding_model():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL, model_kwargs = {"device": "cpu"})

def create_llm_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    hf_pipeline = pipeline(
            task="text-generation",
            model = LLM_MODEL,
            tokenizer = tokenizer,
            trust_remote_code = True,
            max_new_tokens=100,
            model_kwargs={
                "device_map": "auto",
                "load_in_8bit": False,
                "max_length": 512,
                "temperature": 0.01,
                "offload_folder" : "offload",
                "torch_dtype":torch.bfloat16,
                }
        )
    return hf_pipeline

embedding = create_embedding_model()
llm = create_llm_model()

# load pdf
pdf_path = "cpd53120.pdf"
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")
texts = text_splitter.split_documents(texts)

vector_db = Chroma.from_documents(documents=texts, embedding = embedding,persist_directory = None)

hf_llm = HuggingFacePipeline(pipeline=llm)
retriever = vector_db.as_retriever(search_kwargs={"k":4})
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",retriever=retriever)

question = "How to save energy in epson printer?"
qa.combine_documents_chain.verbose = True
qa.return_source_documents = True
qa({"query":question})
