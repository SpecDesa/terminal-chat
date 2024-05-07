from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain 
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv

import argparse

load_dotenv()

parser = argparse.ArgumentParser()
# parser.add_argument()
args = parser.parse_args()



chat = ChatOpenAI()
memory = ConversationBufferMemory(
        chat_memory=FileChatMessageHistory("messages.json"),
        memory_key="messages", 
        return_messages=True)

chat_prompt = ChatPromptTemplate(
        input_variables=["content", "messages"],
        messages=[
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{content}")
            ]
        )

chain = LLMChain(
        llm=chat,
        prompt=chat_prompt,
        memory=memory
        )

while True:
    content = input(">> ")
    result = chain({"content": content})

    print(result["text"])
