import chromadb # my data that I use
from google import genai #this for Embed the question
from groq import Groq # this for Generating stage
# ++++++
# I used the langchain libs in Hybrid search
from langchain_community.retrievers import BM25Retriever as BM25 # for word search
from langchain.retrievers import EnsembleRetriever as ensambl 
from langchain_core.documents import Document 
from langchain_core.retrievers import BaseRetriever
from typing import List
# ++++++
import gradio as grad # to make a public link 
import re 
import os # used in geting keys 
import time
from dotenv import load_dotenv # To hide my API_Keys
load_dotenv()


class chrom:
    def __init__(self):
       pass
    def setup(self):
        chrom_client=chromadb.PersistentClient(path=r"A:\Bishoy\books\vscode\chatpopa\vector")
        collection_text=chrom_client.get_collection(name="final_stable_step")
        return collection_text

class models:
    def __init__(self):
        pass
    def google_model_embed(self,text):
        embed_model=genai.Client(api_key=os.getenv("google_API"))
        word2vec=embed_model.models.embed_content(
            model="models/gemini-embedding-2",
            contents=text,
            config=genai.types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        return word2vec
    
    def groq_model_text(self,full_prompt):
        gr_api=Groq(api_key=os.getenv("groq_API"))
        response=gr_api.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role":"system", "content":"أنت لديك معلومات تاريخية عن الكنيسة القبطية الأرثوذكسية و مطلوب منك شرحها بأسلوب مبسط وتحكي ها كقصة مسلية"},
                {"role":"user", "content":full_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
            
        )
        clean_text=re.sub(r'<think>.*?</think>','',response.choices[0].message.content, flags=re.DOTALL).strip()
        return clean_text
    
class StaticVectorRetriever(BaseRetriever):
    docs: List[Document]
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self.docs


class master:
    def __init__(self):
        self.coll=chrom()
        self.model=models()
        
    def hybridsearch(self,message, docs):
        pure_texts = docs[0] if docs else []
    
        # تحويل النصوص العادية إلى كائنات Document المطلوبة للمكتبة
        langchain_docs = [Document(page_content=text) for text in pure_texts]
    
        # تهيئة مسترجع الكلمات المفتاحية وتحديد أعلى 3
        bm_25 = BM25.from_documents(langchain_docs)
        bm_25.k = 3 
    
        # تهيئة مسترجع المتجهات الاستاتيكي بناءً على ترتيب كروما المرجع
        vector_retriever = StaticVectorRetriever(docs=langchain_docs)
    
        # دمج المسترجعين بـ EnsembleRetriever لتطبيق الـ RRF تلقائياً بالخلفية
        assambl = ensambl(
        retrievers=[bm_25, vector_retriever],
        weights=[0.4, 0.6]
        )
        # تشغيل البحث الهجين ممرراً نص السؤال
        final_docs = assambl.invoke(message)
    
        # إرجاع قائمة نصوص صافية لتتوافق مع الـ join في دالة الاستعلام لديك
        return [doc.page_content for doc in final_docs]
    
    def img_show(self,raw_data):
        image=raw_data['metadatas'][0][0]
        img=image.get('image_url')
        if img is not None:
            time.sleep(0.2)
            return img
        else:
            return None

    def llm_send(self,message, history):
        q=self.coll.setup()
        request=self.model.google_model_embed(message)
        embedding = request.embeddings[0].values

        raw=q.query(
            query_embeddings=[embedding],
            n_results=2,
            include=["documents","embeddings","metadatas"]
        )

        result=self.hybridsearch(message, raw["documents"])
        context="\n".join(result)
        full_prompt = f"""أنت مساعد متخصص في التاريخ الكنسي.
أجب بالعامية المصرية  بناءً على السياق:

السياق:
{context}

السؤال: {message}

- لو المعلومة مش موجودة قول "المعلومة غير متوفرة في المصدر"
- كن دقيقاً ومختصراً
-حافظ على شكل و كتابة الاسم او التاريخ المطابق للمصدر ولا تقم بترجمة أي اسم للعربية 
-دقق فى الأخطاء الإملائية قبل عرضها 
-راجع النص للتأكد من خلوه من الأخطاء
-*تأكد من إجابتك مرة أخرى قبل الإرسال*


الإجابة:"""
       
        
        # if history:
        #     messages = history + [{"role": "user", "content": full_prompt}]
        # else:
        #     messages=full_prompt
        
        messages=full_prompt
        response=self.model.groq_model_text(messages)
        current_text=" "
        for character in response:
            current_text += character
            time.sleep(0.05) 
            yield current_text
        im=self.img_show(raw)
        if im:
            yield (im,context)



# ++++++++++++++
one=master()




# +++++++++++++++
with grad.Blocks(title="المساعد الكنسي", theme=grad.themes.Soft()) as demo:
    grad.Markdown("# ⛪ المساعد الكنسي")
    
    grad.ChatInterface(
        fn=one.llm_send,
        chatbot=grad.Chatbot(height=500, rtl=True),
        textbox=grad.Textbox(placeholder="اسأل عن التاريخ الكنسي...", rtl=True),
    )

demo.launch(share=True)