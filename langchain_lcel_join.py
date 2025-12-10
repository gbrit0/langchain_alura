from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_classic.globals import set_debug

from pydantic import BaseModel, Field

from dotenv import load_dotenv
from operator import itemgetter
import os

set_debug(True) # Detalhes da execução são impressos no terminal

load_dotenv(override=True)
apikey = os.environ['OPENAI_API_KEY']

class Destino(BaseModel):
    cidade: str = Field("Cidade a visitar")
    motivo: str = Field("Motivo pelo qual é interessante visitar a cidade")

parser = JsonOutputParser(pydantic_object=Destino)

llm = ChatOpenAI(
    api_key=apikey, 
    model='gpt-3.5-turbo', 
    temperature=0.5
)

modelo_cidade = PromptTemplate(
    template="""Sugira uma cidade dado o meu interesse por {interesse}.
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

modelo_final = ChatPromptTemplate.from_messages(
    [
        ("ai", "Sugestão de viagem para a cidade: {cidade}"),
        ("ai", "Restaurantes que você não pode perder: {restaurantes}"),
        ("ai", "Atividades culturais recomendadas: {locais_culturais}"),
        ("system", "Combine as informalçies anteriores em 2 parágrafos coerentes."),
    ]
)

parte1 = modelo_cidade | llm | parser # Pegar o modelo de prompt, passar para o llm e depois para o parser
parte2 = modelo_restaurantes | llm | StrOutputParser()  # Usar um parser simples que converte a saída em string 
parte3 = modelo_cultural | llm | StrOutputParser()  # Usar um parser simples que converte a saída em string 
parte4 = modelo_final | llm | StrOutputParser()

cadeia = (
    parte1 | 
    {
        "restaurantes": parte2, 
        "locais_culturais": parte3,
        "cidade": itemgetter("cidade") # Pega a cidade do resultado da parte 1
    }
    | parte4
)

resultado = cadeia.invoke({"interesse": "praias"})
print(resultado)
