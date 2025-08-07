from fastapi import FastAPI
app=FastAPI()

@app.get("/go/{id}")
async def root(id:int):
    return {"item_id": id%10 +id//10}

@app.get("/g/{id}")
async def root(id:str):
    return [id]+["Rick", "Morty"]

from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


# app = FastAPI()


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}



@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}