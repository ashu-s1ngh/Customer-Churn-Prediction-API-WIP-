End-to-End Customer Churn Prediction API
Project Status: This is a complete, end-to-end MLOps project. The machine learning model, the FastAPI application, the multi-stage Docker container, and the CI/CD pipeline are all 100% complete and functional.

The final deployment to a live URL is currently paused. The 750MB optimized Docker image exceeds the 500MB free-tier storage limit on all major cloud platforms (AWS, GCP, Azure). The project is ready for deployment the moment a decision is made to either pay the (minimal) storage cost or re-architect the model to fit within the free tier.

This project demonstrates the complete lifecycle of a machine learning model, from data cleaning and training in a notebook to a fully containerized, production-ready API with an automated CI/CD pipeline.

🚀 Features
ML Model: A RandomForestClassifier trained on the Telco Customer Churn dataset, achieving a ROC AUC of 0.84.

FastAPI Backend: A high-performance API built with FastAPI to serve the model.

Data Validation: Uses Pydantic to enforce a strict, production-ready schema for all incoming prediction requests.

Fully Containerized: A Dockerfile with a multi-stage build process to create an optimized, (relatively) lightweight production image.

CI/CD Pipeline: A complete, production-grade CI/CD workflow using GitHub Actions that automatically:

Builds the Docker image.

Securely logs into a cloud container registry.

Pushes the new image to the registry.

Updates the cloud function (e.g., AWS Lambda) to deploy the new image.

🛠 Tech Stack
Backend: Python, FastAPI, Pydantic

ML Stack: Scikit-learn, Pandas, Joblib

DevOps & Deployment: Docker, GitHub Actions, AWS ECR, AWS Lambda, AWS IAM, Mangum (for AWS Lambda)

📂 Project Structure
Customer-Churn-Prediction-API/
│
├── .github/workflows/         # CI/CD pipeline configuration
│   └── deploy.yml
│
├── .dockerignore              # Ignore files in Docker build
├── .gitignore                 # Ignore files in Git
├── churn_model.joblib         # The exported, trained model
├── Dockerfile                 # Multi-stage Docker build
├── main.py                    # The FastAPI application
├── preprocessor_pipeline.joblib # The exported Scikit-learn preprocessor
├── README.md                  # You are here!
├── requirements.txt           # Python dependencies
└── training/
    └── Model_Training.ipynb   # (Or .py) The original notebook for training
1. The Model: From CSV to joblib
The model and preprocessor were built from scratch. The full training script can be found in the training/ directory.

Process:
Load Data: Ingested the WA_Fn-UseC_-Telco-Customer-Churn.csv dataset.

Data Cleaning:

Identified that TotalCharges was incorrectly loaded as an object type.

Converted TotalCharges to numeric, coercing errors which revealed 11 missing values.

Filled these missing TotalCharges (for 0 tenure customers) with 0.

Dropped the original TotalCharges and customerID columns.

Feature Engineering:

Created a ServiceCount feature by summing all binary ('Yes'/'No') internet service columns.

Created Monthly_Charges_per_Service to find a more normalized cost feature.

Preprocessing:

A ColumnTransformer was built to apply StandardScaler to numerical features and OneHotEncoder to categorical features.

This entire preprocessing pipeline was saved as preprocessor_pipeline.joblib.

Training:

A RandomForestClassifier was trained on the processed data.

The model achieved a ROC AUC score of 0.84 and an F1-Score of 0.62 for the 'Churn' class.

The final trained model was saved as churn_model.joblib.

2. The API: Serving the Model
The main.py file creates a production-ready API using FastAPI.

Load: On startup, the API loads churn_model.joblib and preprocessor_pipeline.joblib into memory.

Validate: A Pydantic model (CustomerInput) defines the exact data types and values (e.g., gender: Literal['Male', 'Female']) that the API will accept, preventing bad data.

Predict: The /predict/ endpoint:

Receives the customer data as JSON.

Converts it to a Pandas DataFrame.

Performs the exact same feature engineering (e.g., ServiceCount) that the model was trained on.

Uses the loaded preprocessor to transform the data.

Uses the loaded model to get a prediction.

Returns the churn probability and a binary is_churner flag.

Deploy: The Mangum library is used to wrap the FastAPI app, making it compatible with a serverless environment like AWS Lambda.

3. The Deployment Story: A $0.00 MLOps Challenge
This project's final phase was a deep dive into real-world MLOps challenges.

Challenge 1: The 1.56GB Image

A standard docker build resulted in a 1.56GB image due to the C-dependencies of pandas and scikit-learn. This is too large for any serverless platform.

Solution: I implemented a multi-stage Dockerfile. This uses a "builder" image to compile the libraries, then copies only the necessary artifacts into a slim python:3.11-slim (or public.ecr.aws/lambda/python:3.11) final image.

Result: The image size was successfully reduced to ~750MB.

Challenge 2: Local Docker Push Errors

My local Windows environment was blocked by a persistent The stub received bad data error, making docker login to AWS impossible.

Solution: I bypassed my local machine entirely by building a professional CI/CD pipeline using GitHub Actions. The .github/workflows/deploy.yml file defines a "robot" that runs on a clean Linux machine in the cloud to build, tag, and push the image.

Challenge 3: The $0.00 vs. 750MB Wall

The CI/CD pipeline successfully built and pushed the 750MB image.

The Final Blocker: The original plan (AWS Lambda + ECR Public) failed because Lambda does not support pulling from ECR Public.

The Pivot: The logical pivot was to use a Private ECR repository.

The Hard Wall: All major cloud platforms (AWS, GCP, Azure) have a free-tier storage limit of 500MB. My 750MB optimized image is still too large.

Final Status
The project is fully functional and ready to be deployed. The final hurdle is to either:

Accept the minimal cloud cost (~$0.10/month) to store the 750MB image in a private registry.

Re-architect the model using libraries lighter than scikit-learn to shrink the image below 500MB.

4. How to Run This Project Locally
You can build and run this entire API on your local machine.

Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Build the Docker image:

Bash

docker build -t churn-api .
Run the container:

Bash

docker run -p 8000:8000 churn-api
Access the API: Your API is now running. Open your browser and go to http://127.0.0.1:8000/docs to see the interactive (Swagger) API documentation and send test predictions.