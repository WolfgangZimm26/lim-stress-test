from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import my_module  # Import your module

#call fast api
app = FastAPI()

class InputData(BaseModel):
    prompt: str

@app.post("/process")
def process_input(data: InputData):
    try:
        result = my_module.process_prompt(data.prompt)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
