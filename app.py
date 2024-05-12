from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import glob
from tqdm import tqdm
import json
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
import pandas as pd
from langchain_community.retrievers import BM25Retriever
import pickle
from langchain.retrievers import EnsembleRetriever
import streamlit as st

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

Query_paraphrase_prompt = """
# Character
You are a proficient assistant specializing in query paraphrasing. Your primary function is to break down complex inquiries into smaller, manageable chunks, if the original query encompasses multi-hop problems.

## Skills
### Skill 1: Paraphrase a query
- Grasp the essence of the query given by the user.
- Retranslate the query in a clear and concise manner. 

### Skill 2: Decompose a multi-hop query
- Identify if a query requires multiple steps or 'hops' to answer, if not return the same question. 
- Break down the multi-hop query into distinct, simple queries.
- Ensure that each smaller query contributes to answering the overall question. 

## response in json format

=====

## Examples:

1. Original Query: "Who starred in the film directed by the director of Inception and what awards did they win?"
   
   Decomposed Queries:
   {
   "question_1": "Who was the director of the movie Inception?", 
   "question_2": "What other films has this director directed?",
   }
   
2. Original Query: "How much does Meta's AI investment increase in 2023 compared to 2022?"

   Decomposed Queries:
   {
   "question_1": "How much Meta's AI investment is in 2022?", 
   "question_2": "How much Meta's AI investment is in 2023?"
   }
   
3. Original Query: "How much does Meta's AI investment is in 2023?"

   Decomposed Queries:
   {
   "question_1": "How much does Meta's AI investment is in 2023?", 
   }
   
=====
## Constraints:
- Stick to the thought-process required to break down and paraphrase questions, avoiding unrelated topics.
- Aim for clarity and precision in your rephrasing and decomposing.
- Structure your response directly, without preamble.
- Maximum 2 decomposed queries should be generated.
- Do not split the query if question is simple.

"""

Hyde_sys_prompt = """
# Character
You're a financial answer guesser. Even without the latest news or a complete knowledge base, your can produce a simple answer utilizing placeholders such as "xxx", "xxx%", "$xxx", and so on for unknown elements.

## Constraints:
- Aim to provide the simple reply, even when the latest information is not readily available.
- The number of the letter of the answer is limited to 30 words.

## Example
#Example1
--------
Question: What will Apple's market value be in 2023?
Reply: Apple's market cap in 2023 is $xx billion.

#Example2
--------
Question: How much does Meta's AI investment increase in 2023 compared to 2022?
Reply: Meta will invested $xx billion in AI in 2023 and $xx billion in 2022, an increase of xx%.

"""

QA_sys_prompt = """
# Character
You're a skilled chatbot, capable of extracting relevant information from retrieved documents. When a user poses a question, you answer it as though you have firsthand knowledge rather than referencing a document.

## Skills
### Skill 1: Answer questions about the retrieved document
- Understand the user's question.
- Analyze the retrieved document to find relevant information.

### Skill 2: Reply when unable to answer
- If the question can't be answered based on the document, respond, "Sorry, I do not have an accurate answer for this."

## Constraints
- Mimic the tone and language used by a chatbot.
- Do not reference any document or outside source in your answers.
- If no accurate answer can be provided, be honest and inform the user.
"""

QA_user_prompt = """
## Reference document
{}

Answer this query: {} Make the answer short and clean.
"""

verifier_sys_prompt = """## Role: Answer verifier

## Goal
You can judge whether the answer is correct or not. 

## Rule
- If the key information predicted answer is same as the ground truth answer, then the answer is correct.
- If the response is "Sorry, I do not have an accurate answer for this.", it means the answer can not be found, then the answer can be treated as correct.

## Output format
{{
"reason": "fill the reason why the predicted answer is wrong (False) or correct (True).", 
"answer": True or False
}}
"""

verifier_user_prompt = """
The question is: {}
Ground truth is: {}
Predicted answer is: {}
"""


qa_full_dataset_name = "QA_dataset_v2"  # Define dataset name
TOP_K = 5

def save_pk(chunks, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(chunks, file)

def load_pk(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

@st.cache_resource
def load_doc_chunks(chunk_size, qa_full_dataset_name):
    
    chunks = []
    file_path = f"./chunks/chunks_{chunk_size}_{qa_full_dataset_name}.pkl" 
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_size//5,
        length_function=len
    )
    try:
        # Check if the chunks file already exists
        chunks = load_pk(file_path)
        print("Loaded chunks from existing file. ", file_path)
    except FileNotFoundError:
        # If the file does not exist, process the PDFs
        chunks = []
        for pdf in tqdm(glob.glob("./docs/*.pdf")):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() if page.extract_text() else ""
            chunks += text_splitter.split_text(text)
        
        # Save the chunks to a file after processing
        save_pk(chunks, file_path)
        print("Saved new chunks to file. ", file_path)
    return chunks

@st.cache_resource
def kb_initialization(model_names, chunk_size):
    
    chunks = []   
    retrievers = []
    for model_name in model_names:
        
        # define retriever saving path
        index_filename = f"faiss_index_cs-{chunk_size}_" + model_name.split("/")[-1]
        index_path = "./faiss/" + index_filename
    
        # define embeddings
        if model_name == "text-embedding-ada-002":
            embeddings = OpenAIEmbeddings(model=model_name)
        elif model_name != "BM25":
            embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs = {'device': 'cuda:0'},encode_kwargs = {'normalize_embeddings': True})

        # load retriever
        if not os.path.exists(index_path):
            if chunks == []:
                chunks = load_doc_chunks(chunk_size, qa_full_dataset_name)
            if model_name == "BM25":
                retriever = BM25Retriever.from_texts(chunks, metadatas=[{"source": 1}] * len(chunks))
                retriever.k = TOP_K
                save_pk(retriever, index_path)
            else:
                faiss_vectorstore = FAISS.from_texts(chunks, embeddings)
                faiss_vectorstore.save_local(index_path)
                retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        else:
            if model_name == "BM25":
                retriever = load_pk(index_path)
                retriever.k = TOP_K
            else:
                faiss_vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        retrievers.append(retriever)
        
    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers, weights=[1/len(retrievers) for _ in retrievers]
    )
    return ensemble_retriever

def get_response_and_evaluation(data_loaded, knowledge_base):
    
    correctness = []
    results = []
    for item in tqdm(data_loaded[:]):

        try:
            pdf = item["filename"]
            ques= [item["question_1"] , item["question_2"], item["question_3"]]
            anss = [item["answer_1"], item["answer_2"], item["answer_3"]]
        except:
            print("Key error, please check. ", item)
            continue
        
        for query, answer in zip(ques, anss):
            
            # QP + Hyde
            decomposed_queries = query_paraphrasing(Query_paraphrase_prompt, query)
            
            docs = []
            for dq in decomposed_queries:
                rfr = gpt_llm(Hyde_sys_prompt, dq)
                docs = knowledge_base.invoke(rfr)
            docs = list(set([docs[i].page_content for i in range(len(docs))]))

            QA_prompt = QA_user_prompt.format("\n".join(docs), query)
            response = gpt_llm(QA_sys_prompt, QA_prompt)
            verified_output, verified_bool = response_evaluation(query, answer, response)
            
            print("======================")
            print(f"Query: {query}")
            print(f"Ans: {answer}")
            print(f"Res: {response}")
            print(f"Correct or not: {verified_bool}")
            print("verified_output: ", verified_output)
            
            correctness.append(verified_bool)
            results.append([pdf, query, answer, response, verified_output, verified_bool])
            
    return correctness, results

def query_paraphrasing(Query_paraphrase_prompt, query):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": Query_paraphrase_prompt},
                {"role": "user", "content":  query}
            ]
        )
    resp = json.loads(response.choices[0].message.content)
    return resp.values()

def response_evaluation(query, answer, response):
    verifier_prompt = verifier_user_prompt.format(query, answer, response)
    verified_output = gpt_llm(verifier_sys_prompt, verifier_prompt)
    verified_bool = verified_output.split('"answer": ')[-1]
    if "True" in verified_bool:
        verified_bool = 1
    else:
        verified_bool = 0
    return verified_output, verified_bool

def post_response_evaluation(df):
    
    responses = df.loc[:, "response"].values
    verified_output = df.loc[:, "verified_output"].values
    
    judged_res = []
    for res, v_o in zip(responses, verified_output):
        verified_bool = v_o.split('"answer": ')[-1]
        if "True" in verified_bool:
            verified_bool = 1
        else:
            verified_bool = 0
        if "Sorry" in res:
            verified_bool = 0
        judged_res.append(verified_bool)
    return judged_res

def gpt_llm(system_prompt, user_prompt):
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        top_p = 0.9
    )
    return response.choices[0].message.content

from PIL import Image
img = Image.open(r".\images.jpeg")
st.set_page_config(page_title="Generative Financial Report Q&A Chatbot", page_icon= img)
st.header("Ask the PDF corpus📄")

filenames = glob.glob("./docs/*pdf")
model_names = ["BM25", "mixedbread-ai/mxbai-embed-large-v1", "BAAI/bge-large-en-v1.5"]
chunks = load_doc_chunks(chunk_size = 500, qa_full_dataset_name = qa_full_dataset_name)
knowledge_base = kb_initialization(model_names, chunk_size  = 500)  # ER

st.info(f'There are {len(filenames)} PDF files in the corpus.', icon="ℹ️")
st.info(f'They are split into {len(chunks)} segments.', icon="ℹ️")

query = st.text_input("Ask your Question about your PDF")
if query:
    
    # Naive RAG
    docs = knowledge_base.invoke(query)
    docs = list(set([docs[i].page_content for i in range(len(docs))]))
    
    # # QP + Hyde
    # decomposed_queries = query_paraphrasing(Query_paraphrase_prompt, query)
    
    # docs = []
    # for dq in decomposed_queries:
    #     rfr = gpt_llm(Hyde_sys_prompt, dq)
    #     docs = knowledge_base.invoke(rfr)
    # docs = list(set([docs[i].page_content for i in range(len(docs))]))

    QA_prompt = QA_user_prompt.format("\n".join(docs), query)
    response = gpt_llm(QA_sys_prompt, QA_prompt)
        
    st.success(response)
    