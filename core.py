# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
# pyrefly: ignore [missing-import]
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model='open-mistral-nemo')
response = model.invoke("Hello, how are you?")
print(response.content)
