import torch
import pickle
import uvicorn
import argparse
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from nlpimdbmoviereviews.constants import NARROW_PREDICTABLE_GENRES, PREDICTABLE_GENRES


parser = argparse.ArgumentParser(
    description="Train NLP Movie Genre classification model"
)
parser.add_argument(
    "--drop-hard-genres",
    action="store_true",
    help="Drop movie genres that are hardly predictable with the only synopsis.",
)
args = parser.parse_args()

GENRE_LIST = NARROW_PREDICTABLE_GENRES if args.drop_hard_genres else PREDICTABLE_GENRES

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
            for genre_name, prediction_score in zip(GENRE_LIST, genre_prediction)
            if prediction_score > 0.5
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
