from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = '../VectorStore/faiss_0'
EMBEDDING_PATH = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "../../LLM_MODELS/llama-2-7b-chat.Q5_K_M.gguf"  #"TheBloke/Llama-2-7B-Chat-GGML"

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
    model=MODEL_PATH,
    model_type="llama",
    max_new_tokens=512,
    config={'context_length': 1024, 'max_new_tokens': 512},
    temperature=0.5
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH,
                                   model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

#Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 3}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )

while True:
    query = input("Enter question: ")
    if query == '0':
        break
    output = qa_chain.invoke(input=query)
    print(output['result'])