"""Create a ConversationalRetrievalChain for question/answering."""
import sys
import chromadb
from langchain.callbacks.base import BaseCallbackHandler
#from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
#from langchain.chains import RetrievalQA

class MyCustomSyncHandler(BaseCallbackHandler):
    def __init__(self, mycallbackfunc):
        self.myfunc = mycallbackfunc
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.myfunc(token)
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}", file=sys.stderr)

def load_chroma_vectorstore(chroma_dir) -> Chroma:
    """Load the Chroma vectorstore."""
    chroma = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
        client_settings=chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_dir,
        ),
        collection_name="vector-db",
    )
    return chroma


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    #manager = AsyncCallbackHandler([])
    question_callbacks = [MyCustomSyncHandler(question_handler)]
    stream_callbacks = [MyCustomSyncHandler(stream_handler)]
    # if tracing:
    #     tracer = LangChainTracer()
    #     tracer.load_default_session()
    #     manager.add_handler(tracer)
    #     question_manager.add_handler(tracer)
    #     stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callbacks=question_callbacks,
    )
    streaming_llm = OpenAI(
        streaming=True,
        callbacks=stream_callbacks,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm,
        prompt=CONDENSE_QUESTION_PROMPT,  # , callback_manager=manager
        verbose=False,
    )
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=QA_PROMPT,  # , callback_manager=manager
        verbose=False,
    )
#    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

    qa = ConversationalRetrievalChain(
#        memory=memory,
#        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
        retriever=vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1}),
#        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        # callback_manager=manager,
    )
    # vectorstore=vectorstore,
    return qa
