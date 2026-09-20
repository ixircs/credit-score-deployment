# Deploy model Credit Score ke SageMaker real-time endpoint.
# Dijalankan di AWS SageMaker Notebook (AWS Academy Learner Lab) setelah folder di-upload.
# Alur:
#   1. Kemas model_credit_score.joblib -> model.tar.gz
#   2. Upload ke bucket S3 default SageMaker
#   3. Buat SKLearnModel (entry_point=inference.py, source_dir=folder ini)
#   4. Deploy real-time endpoint (ml.m5.large) -> 'credit-score-endpoint'
#   5. Smoke test: invoke endpoint dengan satu contoh data
# Jalankan: cd ke folder ini, lalu `python deploy_endpoint.py`.
# Endpoint ml.m5.large berbiaya per jam; hapus setelah selesai: python cleanup_endpoint.py
import json
import tarfile

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ---- KONFIG (boleh diedit) ----------------------------------------------
ENDPOINT_NAME = "credit-score-endpoint"
MODEL_FILE = "model_credit_score.joblib"
MODEL_TAR = "model.tar.gz"
MODEL_S3_KEY = "credit-score/model.tar.gz"

REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.4-2"   # versi container scikit-learn SageMaker
# -------------------------------------------------------------------------


def get_lab_role_arn() -> str:
    # Ambil ARN LabRole (AWS Academy Learner Lab).
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def package_model():
    # Kemas model.joblib menjadi model.tar.gz (arcname harus di root tar).
    print(f"[1/5] Mengemas {MODEL_FILE} -> {MODEL_TAR}")
    with tarfile.open(MODEL_TAR, "w:gz") as tar:
        tar.add(MODEL_FILE, arcname=MODEL_FILE)


def upload_model(sm_session) -> str:
    # Upload model.tar.gz ke bucket default SageMaker; kembalikan URI S3.
    bucket = sm_session.default_bucket()
    print(f"[2/5] Upload {MODEL_TAR} -> s3://{bucket}/{MODEL_S3_KEY}")
    boto3.client("s3").upload_file(MODEL_TAR, bucket, MODEL_S3_KEY)
    return f"s3://{bucket}/{MODEL_S3_KEY}"


def main():
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn = get_lab_role_arn()

    package_model()
    model_s3_uri = upload_model(sm_session)

    print(f"[3/5] Membuat SKLearnModel")
    print(f"      Role     : {role_arn}")
    print(f"      Model URI: {model_s3_uri}")
    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        source_dir=".",                 # ikut inference.py, pre_processing.py, requirements.txt
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )

    print(f"[4/5] Deploy endpoint '{ENDPOINT_NAME}'...")
    model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    print("[5/5] Smoke test endpoint...")
    sample = {
        "instances": [{
            "Month": "January", "Age": 50, "Occupation": "Lawyer",
            "Annual_Income": 159560.76, "Monthly_Inhand_Salary": 13580.73,
            "Num_Bank_Accounts": 3, "Num_Credit_Card": 1, "Interest_Rate": 6,
            "Num_of_Loan": 1, "Delay_from_due_date": 2, "Num_of_Delayed_Payment": 12,
            "Changed_Credit_Limit": 6.67, "Num_Credit_Inquiries": 2, "Credit_Mix": "Good",
            "Outstanding_Debt": 1154.59, "Credit_Utilization_Ratio": 34.16,
            "Credit_History_Age": 304, "Payment_of_Min_Amount": "No",
            "Total_EMI_per_month": 130.04, "Amount_invested_monthly": 1238.04,
            "Payment_Behaviour": "Low_spent_Large_value_payments", "Monthly_Balance": 259.99,
        }]
    }
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(sample),
    )
    print("Respons:", resp["Body"].read().decode("utf-8"))
    print(f"\nEndpoint '{ENDPOINT_NAME}' LIVE di {REGION}.")
    print("Jangan lupa hapus setelah selesai: python cleanup_endpoint.py")


if __name__ == "__main__":
    main()
