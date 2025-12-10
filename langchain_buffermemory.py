from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains import ConversationChain
from langchain_classic.globals import set_debug
from langchain_classic.memory import ConversationBufferMemory

from dotenv import load_dotenv
from operator import itemgetter
import os

set_debug(True) # Detalhes da execução são impressos no terminal

load_dotenv(override=True)
apikey = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(
    api_key=apikey, 
    model='gpt-3.5-turbo', 
    temperature=0.5
)

mensagens = [
    "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
    "Qual é o melhor período do ano para visitar em termos de clima?",
    "Quais tipos de atividades ao ar livre estão disponíveis?",
    "Alguma sugestão de acomodação eco-friendly por lá?",
    "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
    "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

memoria = ConversationBufferMemory()

conversation = ConversationChain(llm=llm,
                                 verbose=True,
                                 memory=memoria)

longa_conversa = ""
for mensagem in mensagens:
    resposta = conversation.predict(input=mensagem)
    print(resposta)