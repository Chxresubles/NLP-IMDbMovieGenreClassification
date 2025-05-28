# Natural Language Processing example - IMDb Movie Genre Classification dataset insights and genre classification
An example natural language processing project classifying movie genre from its IMDb synopsis using the [IMDb Movie Genre Classification Dataset](https://www.kaggle.com/datasets/adilshamim8/nlp-task/data).


## Get started
To get started, download the splited [IMDb Movie Genre Classification Dataset](https://www.kaggle.com/datasets/adilshamim8/nlp-task/data) ZIP file and extract its content to a folder named `data` in the root directory of the project.


## Data insights
The notebook `data_insights.ipynb` loads the data from CSV files, displays graphs and descriptions of the data.

1. Install the required packages
```console
pip install -r requirements.txt
```

2. Run the `data_insights.ipynb`


## Model training and validation
The scripts contained in the `scripts` contain the necessary code to train, validate and run the inference of the RoBERTa model.
It uses the local module `nlpimdbmoviereviews` containing the project-specific source code.

1. Install the `nlpimdbmoviereviews` module
```console
pip install .
```

2. Train the model
```console
python ./scripts/train.py
```
The scripts saves the model and the train/test metrics in the `output` folder.

3. Validate the model on new data
```console
python ./scripts/validation.py --data-path ./data/new_validation_data
```


## Deploy API

### Run locally

1. Install the `nlpimdbmoviereviews` module
```console
pip install .
```

2. Deploy the API
```console
python ./scripts/score.py
```

3. Send a test request to the API.
```console
python ./scripts/test_api.py
```

### Using Docker
The `Dockerfile` contains a minimal environment to deploy an API on port 8000.

1. Build the `Dockerfile`
```console
docker build . -t nlpimdbmoviereviews-api
```

You can set the argument `cpu_only` to 0 or 1 to build the Docker image using pytorch on the CPU only. By default, `cpu_only` is set to 0 and PyTorch will only be installed with CUDA if an NVIDIA GPU is available.
```console
docker build . -t nlpimdbmoviereviews-api --build-arg cpu_only=1
```

2. Run the Docker image locally
```console
docker run --rm -p 8000:8000 nlpimdbmoviereviews-api
```
You can also use the `-d` option to run the Docker image in the background.

If the machine you are using contains an NVIDIA GPU, use the following command to use CUDA to run the inference of the model
```console
docker run --gpus all --rm -p 8000:8000 nlpimdbmoviereviews-api
```

3. Send a test request to the API.
```console
python ./scripts/test_api.py
```
