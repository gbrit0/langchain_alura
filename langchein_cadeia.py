from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import LLMChain, SimpleSequentialChain

from dotenv import load_dotenv
import os

load_dotenv(override=True)
apikey = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(
    api_key=apikey, 
    model='gpt-3.5-turbo', 
    temperature=0.5
)

modelo_cidade = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado o meu interesse por {interesse}."
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre moradores locais em {cidade}."
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}."
)

cadeia_cidade = LLMChain(prompt=modelo_cidade, llm=llm)
cadeia_reataurantes = LLMChain(prompt=modelo_restaurantes, llm=llm)
cadeia_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

cadeia = SimpleSequentialChain(chains=[cadeia_cidade, cadeia_reataurantes, cadeia_cultural])