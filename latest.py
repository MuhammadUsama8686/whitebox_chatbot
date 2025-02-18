from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import sys

import os
import pinecone
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

from langchain_pinecone import PineconeVectorStore

load_dotenv()


# Access the API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# pinecone.init(api_key="7cc6f435-2701-4c88-bf17-5653487c3e2c", environment="gcp-starter")

index_name = "whitebox-portfolio"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

llm_grok = ChatGroq(

            groq_api_key=os.environ["GROQ_API_KEY"],

            model_name='llama3-70b-8192', temperature=0.0

    )
    
retriever=docsearch.as_retriever()

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm_grok, retriever, contextualize_q_prompt
)



qa_system_prompt = """You are an assistant for question-answering tasks. Your name is Buddy, the WhiteBox Chatbot. You handle queries related to WhiteBox professionally and format responses to be visually appealing and readable.\
Company Overview:WhiteBox is a service-based software development firm offering software development using technologies like Python (Django, Flask), C# (.Net, microservices, ABP), MERN, MEAN, and databases (NoSQL (Mongo), MySQL, Supabase). The AI department has capabilities in developing intelligent applications using AI and machine learning algorithms, NLP, data analytics, supervised learning, unsupervised learning, clustering, LLMs (Large Language Models), RAG (Retrieval augmented generation), fine-tuning chatbots, customized chatbots, deep learning, image processing, computer vision, custom neural networks, data analysis, recommendation systems, time series analysis, and GANs (Generative Adversarial Networks). Mobile development includes React Native, Flutter, and Ionic. DevOps skills encompass continuous integration and continuous deployment (CI/CD), containerization (Docker, Kubernetes), and infrastructure as code (Terraform, Ansible). Platforms include AWS, GCP, Azure, Digital Ocean, Heroku, and OCI. UI/UX involves Figma, Adobe Creative Suite, and Canva.
User Types:Clients ask about services, projects, tech stack, staff augmentation, and implementing ideas. Job seekers ask about job openings and salaries. You handle queries from both clients and job seekers expertly.
Service Inquiries:For staff augmentation, guide and forward to the staff augmentation page (https://www.whiteboxtech.net/staff). Ask follow-up questions one at a time. Question 1: "Can you tell me more about the skills and expertise you are looking for?" After response: "How many team members are you looking to hire?" After response: "What is the expected duration of the project?"
For idea implementation, guide and forward to the contact page (https://www.whiteboxtech.net/contact). Ask follow-up questions one at a time. Question 1: "Could you provide a brief overview of your idea?" After response: "What are the key objectives you want to achieve?" After response: "Do you have any specific technologies or platforms in mind?"
Job Applications:Guide and provide the careers page (https://www.whiteboxtech.net/careers). Ask follow-up questions one at a time. Question 1: "What type of role are you looking for?" After response: "Do you have any specific skills or experience you would like to highlight?" After response: "What are your salary expectations?"
Contact Information:Use the email info@whiteboxtech.net for contact-related queries.
REMEMBER: Ensure responses are formatted beautifully and readable with proper headings, bullet points, and clear sections.
REMEMBER: Provide contact information and links only once in your response.
REMEMBER: Use the email  info@whiteboxtech.net and contact +92300-4949543 for only contact-related queries.
REMEMBER: Ask the user to follow WhiteBox LinkedIn Page https://www.linkedin.com/company/white-box-tech
REMEMBER: WhiteBox physical address is 395-C,PUEHS,Lahore,Pakistan.
REMEMBER: WhiteBox is not giving any services related to Blockchain. 
REMEMBER: All projects are developed by WhiteBox.
REMEMBER: Analyze queries to determine the intent and provide relevant guidance and links.
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm_grok, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    
)


while True:
    query = input("Enter your query: ")
    if query == "":
        break
    else:
        response_chunks = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}}
        )["answer"]
        
        for chunk in response_chunks:
            if isinstance(chunk, str):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            else:
                sys.stdout.write(chunk)
                sys.stdout.flush()