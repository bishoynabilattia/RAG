# -*- coding: utf-8 -*-
from google import genai
import arabic_reshaper
from bidi.algorithm import get_display
import chromadb
import re #لتنظيف النصوص
 
import os
from dotenv import load_dotenv
load_dotenv()
# ++++++++++++++++=============
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter as splitter

client = chromadb.PersistentClient(path="A:\Bishoy\\books\\vscode\chatpopa\\vector")

def reshape(text):
    res = arabic_reshaper.reshape(text)
    finaltext = get_display(res)
    return finaltext

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


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text

def embed_pdf_by_google ( book_name,doc_id, document,meta):
    adcol=client.get_collection(name="final_stable_step")
    adcol.add(
        ids=f"{book_name}_{doc_id}_{meta['bishop_id']}",
        documents=[document],
        metadatas=[meta],
        embeddings=[get_google_embedding(document)]
    )


def read_pdf(file):
    reader=PdfReader(file)
    full_text=""

    for page in reader.pages:
        text=page.extract_text()
        if text:
            full_text+=text+"\n"
    return full_text


def run(chunks,book_name,meta):
    for i, chunk in enumerate(chunks):
        embed_pdf_by_google(book_name, i, chunk, meta)
        print(f"{i+1} has been added for four ")
        
embedd=splitter(chunk_size=1500, chunk_overlap=170, length_function=len,separators=["\n\n", "\n", " ", ""])#التقسيم العادي



pdf_file=r"A:\raw data\تاريخ الايبارشية\المراجع\4الآباء الأساقفة.pdf"
book_name="الأباء الأساقفة الأربعة"

print("*"*50)
text=read_pdf(pdf_file)
retext=clean_text(text)
chunks=embedd.split_text(retext) #list of minichunks
print(reshape(f"عدد الـ لChunks الكلي: {len(chunks)}"))



# ---------to delete-----------
#  client.delete_collection("final_stable_step")

# ________________DATA________________
adcol=client.get_or_create_collection(name="final_stable_step")

meta = {
    "bishop_id": 4,
    "bishop_name": "الأباء الأساقفة الأربعة",
    "category": "Biography",
    "diocese": "سمالوط",
    "image_url":r"A:\raw data\تاريخ الايبارشية\الصور\الاربعة4.png",
    "source_book": "تاريخ إيبارشية سمالوط"
}

# _____________Execution_____________
run(chunks, book_name,meta)
print(f"the total number is : {adcol.count()}")
