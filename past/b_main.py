from fastapi import FastAPI
from b_chat_with_rag import chat
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class Item(BaseModel):
    input: str

@app.post("/")
async def root(item: Item):
    print(item)
    json = chat(item.input)
    print(json)
    return json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# pip install fastapi uvicorn
# pip install python-dotenv