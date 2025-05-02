# mlops-ead

## Python version
This code was created with python version 3.11.6.
Use pyenv to install this version.

To run the exercises, follow the instructions bellow

# Create Environment with venv

Create the virtual environment:

```
python -m venv mlops-ead-env
```
## Python version
This code was created with python version 3.11.6.
Use pyenv to install this version.
Activate the virtual environment

- Windows:
```
mlops-ead-env\Scripts\activate
```

- Linux/macOS:
```
source mlops-ead-env/bin/activate
```

## Install the requirements

```
pip install -r requirements.txt
```

## Deactivate the environment

```
deactivate
```

# Create Environment with conda

```
conda create -f environment.yml
```

## Activate environment
```
conda activate mlops-ead-env
```

## Deactivate environment
```
conda deactivate
```

# API

## Execution

Para executar a API na porta 8000, é necessário executar o seguinte comando no terminal:
```
uvicorn src.main:app --reload
```

## Documentation

A documentação pode ser encontrada na rota */docs*.

## Build docker image

```
docker build --no-cache -t model_api .
```

## List images

```
docker image list
```

## Running image

### Add permition to script (one time only)
```
chmod +x run_api.sh
```

### Running script
```
./run_api.sh
```