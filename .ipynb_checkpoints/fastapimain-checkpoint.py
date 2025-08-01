from fastapi import FastAPI
app=FastAPI()

@app.get("/go/{id}")
async def root(id:int):
    return {"item_id": id%10 +id//10}

@app.get("/g/{id}")
async def root(id:str):
    return [id]+["Rick", "Morty"]