# Titanic API

## Quick Presentation

This package is meant to provide a deployable API that predicts whether a person will survive a trip given their information (ticket, price, class, sex, etc). It offers the user the choice between different models.

## How to install and launch the server

Thanks to Docker, the installation is quite straightforward. 

### Docker installation

If you do not already have Docker on your machine, please follow the instructions [here](https://docs.docker.com/desktop/).

Once Docker is installed, make sure to launch the Docker Engine: 
- On Windows you will need to open the Docker Desktop app. Alternatively you can open a powershell terminal and run: ```start "C:\path\to\docker\Docker Desktop.exe" ```
- On Linux, you can run: ```sudo systemctl start docker```

### Create the docker image

Open a bash terminal and make sure you are at the root of the package. Run the following command: 

```docker build -t titanic-ml-api ./```

### Create the volume

The first time you are running the image, you will need to create a volume. This volume is meant for data persistance across containers. Run: 

``docker volume create titanic-models-vol``

### Run it

To run in a dev environment: 

``docker run -it -p 8000:8000 -v titanic-models-vol:/app/models/ titanic-ml-api``

Description:
- ``-it`` flag will allow to print the logs and STDOUT on the console.
- ``-p`` flag declares a port for the API.
- ``-v`` flag declares a volume ("where it is stored on the host machine":"where it is stored in the docker app").

## How to use the API

Here are the different routes at users' disposition. Note that the this example is meant for development. Once the API is deployed you will replace http://127.0.0.1:8000/ by https://your_website_url/ .
- http://127.0.0.1:8000/ (GET)
    - You can use this to check that your API server is live.
    - Returns: 
        ```json
        {
            "message": "Welcome to our MLOps API."
        }
        ```
- http://127.0.0.1:8000/model/default (GET)
    - Returns the default model and its accuracy.
    - Returns:
        ```json
        { 
            "model": "model type (str)", 
            "accuracy": "model accuracy (int)" 
        }
        ```
- http://127.0.0.1:8000/model/list (GET)
    - Returns the list of all usable models as well as their accuracy.
    - Returns: 
        ```json
        { 
            "1st model": {"accuracy": "1st model's accuracy (int)"},
            "2nd model": {"accuracy": "2nd model's accuracy (int)"}
        }
        ```
- http://127.0.0.1:8000/features (GET)
    - Allows the user to get the list of features and their types.
    - Returns: 
        ```json
        {
        "features in order": """list of feature names (list[str])""",
        "types": """list of types (list[int])"""
        }
        ```
- http://127.0.0.1:8000/predict (GET)
    - Makes a prediction for as many data points as needed.
    - Parameters to provide in the body: 
        ```json
        {
        "model": "model type (see the list of models in /model/list/)",
        "features": {
            "1st feature's name": ["1st data point", "2nd data point", "etc"],
            "2nd feature's name": ["1st data point", "2nd data point", "etc"],
            "etc": "etc"
            }
        }
        ```
    - Returns:
        ```json
        {
        "prediction": ["1st data point prediction", "2nd data point prediction", "etc"]      
        }
        ```

