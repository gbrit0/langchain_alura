from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv(override=True)
apikey = os.environ['OPENAI_API_KEY']

numero_de_dias = 3
numero_de_criancas = 2
atividade = "praia"

modelo_do_prompt = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {dias} dias, para uma família com {criancas} crianças, que gostam de {atividade}."
)

prompt = modelo_do_prompt.format(
    dias=numero_de_dias,
    criancas=numero_de_criancas,
    atividade=atividade
)

print(prompt)


llm = ChatOpenAI(
    api_key=apikey, 
    model='gpt-3.5-turbo', 
    temperature=0.5
)

resposta = llm.invoke(prompt)

print(resposta.content)