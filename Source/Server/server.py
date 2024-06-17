from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import asyncio
from fastapi import FastAPI, Request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config
from starlette.responses import JSONResponse

config = Config()
config.bind = ["localhost:8888"]
app = FastAPI()

DB_FAISS_PATH = '../VectorStore/faiss_0'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])

llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_type="llama",
    max_new_tokens=512,
    config={'context_length': 1024, 'max_new_tokens': 512},
    temperature=0.5
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

#Retrieval QA Chain
chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 3}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )

@app.post("/message")
async def root(request: Request):
    data = await request.json()
    message = data['message']
    print("Input to LLM ---> ", message)

    res = await chain.ainvoke(input={'query': message})
    answer = res["result"]
    sources = res["source_documents"]
    print('Response from LLM ---> ', answer)

    return JSONResponse(content={'status': 200, 'response': answer})


asyncio.run(serve(app, config))