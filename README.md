# Health Prediction Web Application

This project is a web application built using Flask that predicts the likelihood of diabetes and stroke based on user input. The application uses machine learning models trained on relevant datasets to make predictions.

## Project Structure


## Files and Directories

- `app.py`: The main Flask application file that defines routes and handles requests.
- `diabetes.csv`: The dataset used for training the diabetes prediction model.
- `diabetes.ipynb`: Jupyter notebook for data analysis and model training for diabetes prediction.
- `model.pkl`: Pickle file containing the trained diabetes prediction model.
- `model.py`: Script for training the diabetes prediction model.
- `pa ass !!!.ipynb`: Jupyter notebook for data analysis and model training for stroke prediction.
- `Procfile`: Configuration file for deploying the application on Heroku.
- `README.md`: This file.
- `requirements.txt`: List of Python dependencies required for the project.
- `runtime.txt`: Specifies the Python runtime version for Heroku deployment.
- `scaler.pkl`: Pickle file containing the scaler used for feature scaling in diabetes prediction.
- `stroke prediction.csv`: The dataset used for training the stroke prediction model.
- `strokenew.pkl`: Pickle file containing the trained stroke prediction model.
- `.vscode/`: Directory containing Visual Studio Code settings.
- `static/`: Directory containing static files (CSS, JS, images).
- `templates/`: Directory containing HTML templates.

## Installation

1. Clone the repository:
    
    git clone https://github.com/Mousam861/Health-Monitoring
    cd health-prediction
    

2. Create a virtual environment and activate it:
   
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    

3. Install the required dependencies:
    
    pip install -r requirements.txt
    

4. Run the Flask application:
    
    python app.py
    

5. Open your web browser and go to `http://127.0.0.1:5000`.




## Usage

### Home Page

The home page provides links to check the risk of stroke and diabetes.

### BMI Checker

Navigate to the BMI Checker page to calculate your Body Mass Index (BMI).

### Diabetes Prediction

Navigate to the Diabetes Prediction page and enter the required details to predict the likelihood of having diabetes.

### Stroke Prediction

Navigate to the Stroke Prediction page and enter the required details to predict the likelihood of having a stroke.

### Contact

Navigate to the Contact page to get in touch with the developers.

## Model Training

### Diabetes Prediction Model

The diabetes prediction model is trained using the `diabetes.ipynb` notebook. The model uses features such as Glucose, Insulin, BMI, and Age to predict the likelihood of diabetes.

### Stroke Prediction Model

The stroke prediction model is trained using the `pa ass !!!.ipynb` notebook. The model uses features such as age, hypertension, heart disease, and others to predict the likelihood of a stroke.

## Deployment

The application can be deployed on Heroku. Ensure that the `Procfile` and `runtime.txt` files are correctly configured.



## Acknowledgements

- The diabetes dataset is sourced from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
- The stroke dataset is sourced from [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset).




