from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser

from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_classic.globals import set_debug


from dotenv import load_dotenv
import os

set_debug(True)

load_dotenv(override=True)
apikey = os.environ['OPENAI_API_KEY']

class Destino(BaseModel):
    cidade = Field("Cidade a visitar")
    motivo = Fiel("Motivo pelo qual é interessante visitar a cidade")

parser = JsonOutputParser(pydantic_object=Destino)

llm = ChatOpenAI(
    api_key=apikey, 
    model='gpt-3.5-turbo', 
    temperature=0.5
)

modelo_cidade = PromptTemplate(
    template="""Sugira uma cidade dado o meu interesse por {interesse}. A sua saída deve ser SOMENTE o nome da cidade.
    {formatacao_de_saida}""",
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parser.get_format_instructions()}
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre moradores locais em {cidade}."
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}."
)

cadeia_cidade = LLMChain(prompt=modelo_cidade, llm=llm) # É possível passar mais de um modelo de llm. Por exemplo usar o groq etc...
cadeia_reataurantes = LLMChain(prompt=modelo_restaurantes, llm=llm)
cadeia_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

cadeia = SimpleSequentialChain(chains=[
    cadeia_cidade
    # , cadeia_reataurantes, 
    # cadeia_cultural
    ], verbose=True)

resultado = cadeia.invoke("praias")
print(resultado)
