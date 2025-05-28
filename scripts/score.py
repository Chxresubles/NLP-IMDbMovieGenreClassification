from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import uvicorn
import torch
from nlpimdbmoviereviews.constants import NARROW_PREDICTABLE_GENRES

# Initialize FastAPI app
app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading tensors on {DEVICE.type}")

# Load the model
try:
    with open("./output/model.pkl", "rb") as f:
        model = pickle.load(f).to(device=DEVICE)
except FileNotFoundError:
    raise Exception("Model files not found")


class InputSchema(BaseModel):
    overview: str


@app.post("/predict")
async def predict(data: InputSchema):
    try:
        # Make prediction
        with torch.no_grad():
            genre_prediction = model(data.overview).cpu().detach().numpy()[0]

        result = data.model_dump() | {
            genre_name: prediction_score.item()
            for genre_name, prediction_score in zip(
                NARROW_PREDICTABLE_GENRES, genre_prediction
            )
            if prediction_score > 0.5
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
