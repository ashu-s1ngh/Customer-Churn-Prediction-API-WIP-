# End-to-End Customer Churn Prediction API

> **Project Status:** This is a complete, end-to-end MLOps project. The machine learning model, the FastAPI application, the multi-stage Docker container, and the CI/CD pipeline are all 100% complete and functional.
>
> **The final deployment to a live URL is currently paused.** The 750MB optimized Docker image exceeds the 500MB free-tier storage limit on all major cloud platforms (AWS, GCP, Azure). The project is ready for deployment the moment a decision is made to either pay the (minimal) storage cost or re-architect the model to fit within the free tier.

This project demonstrates the complete lifecycle of a machine learning model, from data cleaning and training in a notebook to a fully containerized, production-ready API with an automated CI/CD pipeline.

---

## 🚀 Features

* **ML Model:** A `RandomForestClassifier` trained on the Telco Customer Churn dataset, achieving a **ROC AUC of 0.84**.
* **FastAPI Backend:** A high-performance API built with `FastAPI` to serve the model.
* **Data Validation:** Uses `Pydantic` to enforce a strict, production-ready schema for all incoming prediction requests.
* **Fully Containerized:** A `Dockerfile` with a **multi-stage build** process to create an optimized, (relatively) lightweight production image.
* **CI/CD Pipeline:** A complete, production-grade CI/CD workflow using `GitHub Actions` that automatically:
    * Builds the Docker image.
    * Securely logs into a cloud container registry.
    * Pushes the new image to the registry.
    * Updates the cloud function (e.g., AWS Lambda) to deploy the new image.

---

## 🛠 Tech Stack

* **Backend:** Python, FastAPI, Pydantic
* **ML Stack:** Scikit-learn, Pandas, Joblib
* **DevOps & Deployment:** Docker, GitHub Actions, AWS ECR, AWS Lambda, AWS IAM, Mangum (for AWS Lambda)

---

## 📂 Project Structure

* `Customer-Churn-Prediction-API/`
    * `.github/`
        * `workflows/`
            * `deploy.yml`
    * `training/`
        * `Model_Training_Script.py`
    * `.dockerignore`
    * `.gitignore`
    * `churn_model.joblib`
    * `Dockerfile`
    * `main.py`
    * `preprocessor_pipeline.joblib`
    * `README.md`
    * `requirements.txt`

---

## 1. The Model: From CSV to `joblib`

The model and preprocessor were built from scratch. The full training script can be found in the `training/` directory.

### Process:
1.  **Load Data:** Ingested the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset.
2.  **Data Cleaning:**
    * Identified that `TotalCharges` was incorrectly loaded as an `object` type.
    * Converted `TotalCharges` to numeric, coercing errors which revealed 11 missing values.
    * Filled these missing `TotalCharges` (for 0 `tenure` customers) with `0`.
    * Dropped the original `TotalCharges` and `customerID` columns.
3.  **Feature Engineering:**
    * Created a `ServiceCount` feature by summing all binary ('Yes'/'No') internet service columns.
    * Created `Monthly_Charges_per_Service` to find a more normalized cost feature.
4.  **Preprocessing:**
    * A `ColumnTransformer` was built to apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
    * This entire preprocessing pipeline was saved as `preprocessor_pipeline.joblib`.
5.  **Training:**
    * A `RandomForestClassifier` was trained on the processed data.
    * The model achieved a **ROC AUC score of 0.84** and an **F1-Score of 0.62** for the 'Churn' class.
    * The final trained model was saved as `churn_model.joblib`.

## 2. The API: Serving the Model

The `main.py` file creates a production-ready API using FastAPI.

* **Load:** On startup, the API loads `churn_model.joblib` and `preprocessor_pipeline.joblib` into memory.
* **Validate:** A `Pydantic` model (`CustomerInput`) defines the *exact* data types and values (e.g., `gender: Literal['Male', 'Female']`) that the API will accept, preventing bad data.
* **Predict:** The `/predict/` endpoint:
    1.  Receives the customer data as JSON.
    2.  Converts it to a Pandas DataFrame.
    3.  Performs the *exact same feature engineering* (e.g., `ServiceCount`) that the model was trained on.
    4.  Uses the loaded `preprocessor` to transform the data.
    5.  Uses the loaded `model` to get a prediction.
    6.  Returns the churn probability and a binary `is_churner` flag.
* **Deploy:** The `Mangum` library is used to wrap the FastAPI app, making it compatible with a serverless environment like AWS Lambda.

## 3. The Deployment Story: A $0.00 MLOps Challenge

This project's final phase was a deep dive into real-world MLOps challenges.

1.  **Challenge 1: The 1.56GB Image**
    * A standard `docker build` resulted in a 1.56GB image due to the C-dependencies of `pandas` and `scikit-learn`. This is too large for any serverless platform.
    * **Solution:** I implemented a **multi-stage Dockerfile**. This uses a "builder" image to compile the libraries, then copies *only* the necessary artifacts into a slim `python:3.11-slim` final image.
    * **Result:** The image size was successfully reduced to **~750MB**.

2.  **Challenge 2: Local Docker Push Errors**
    * My local Windows environment was blocked by a persistent `The stub received bad data` error, making `docker login` to AWS impossible.
    * **Solution:** I bypassed my local machine entirely by building a professional **CI/CD pipeline using GitHub Actions**. The `.github/workflows/deploy.yml` file defines a "robot" that runs on a clean Linux machine in the cloud to build, tag, and push the image.

3.  **Challenge 3: The $0.00 vs. 750MB Wall**
    * The CI/CD pipeline successfully built and pushed the 750MB image.
    * **The Final Blocker:** The original plan (AWS Lambda + ECR Public) failed because **Lambda does not support pulling from ECR Public**.
    * **The Pivot:** The logical pivot was to use a **Private ECR repository**.
    * **The Hard Wall:** All major cloud platforms (AWS, GCP, Azure) have a free-tier storage limit of **500MB**. My 750MB optimized image is still too large.

### Final Status

The project is fully functional and ready to be deployed. The final hurdle is to either:
1.  Accept the minimal cloud cost (~$0.10/month) to store the 750MB image in a private registry.
2.  Re-architect the model using libraries lighter than `scikit-learn` to shrink the image below 500MB.

## 4. How to Run This Project Locally

You can build and run this entire API on your local machine.

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/ashu-s1ngh/Customer-Churn-Prediction-API-WIP-.git
    cd Customer-Churn-Prediction-API-WIP-
    ```

2.  **Build the Docker image:**
    ```sh
    docker build -t churn-api .
    ```

3.  **Run the container:**
    ```sh
    docker run -p 8000:8000 churn-api
    ```

4.  **Access the API:**
    Your API is now running. Open your browser and go to `http://127.0.0.1:8000/docs` to see the interactive (Swagger) API documentation and send test predictions.

**Terminal view:**
<img width="1611" height="249" alt="image" src="https://github.com/user-attachments/assets/a09780d6-4cf7-41ae-8983-e3c57ef52982" />

**Example entry:**
<img width="1899" height="902" alt="image" src="https://github.com/user-attachments/assets/1330d7cc-fd10-4112-bfb0-17ad84d42397" />

**Prediction:**
<img width="1899" height="913" alt="image" src="https://github.com/user-attachments/assets/2a6e3093-9200-4f82-a3c1-a9f1548e81fe" />
