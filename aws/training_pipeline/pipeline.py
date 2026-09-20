# Orchestrator SageMaker Pipeline (2a) — training ML di cloud AWS.
# Merangkai 4 tahap (Ingest -> Preprocess -> Train -> Evaluate) sebagai SageMaker
# Pipeline dan menjalankannya dalam local mode (di dalam Notebook instance).
# Jalankan dari terminal Notebook: cd ke folder ini, lalu `python pipeline.py`.
# Prasyarat: gunakan SageMaker Notebook instance (bukan Studio) karena local mode
# butuh Docker. Taruh data_A.csv + semua .py ini di folder yang sama.
import os

import sagemaker
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.pipeline import Pipeline

# ---- Konfigurasi -------------------------------------------------------------
local_session = LocalPipelineSession()

try:
    role = sagemaker.get_execution_role()
except Exception:
    role = "LabRole"          # fallback AWS Academy Learner Lab

INSTANCE_TYPE = "local"       # jalankan pipeline di notebook (local mode)
FRAMEWORK_VERSION = "1.2-1"   # container scikit-learn SageMaker

# Folder kerja = tempat file ini berada (mis. ~/SageMaker/UAS). data_A.csv harus di sini.
BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ["ingested", "train", "test", "eval"]:
    os.makedirs(f"{BASE}/{sub}", exist_ok=True)

OUT = f"file://{BASE}"        # URI output lokal untuk step SageMaker

# ---- Processor & Estimator ---------------------------------------------------
processor = SKLearnProcessor(
    framework_version=FRAMEWORK_VERSION,
    role=role,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    sagemaker_session=local_session,
)

estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type=INSTANCE_TYPE,
    framework_version=FRAMEWORK_VERSION,
    sagemaker_session=local_session,
)

# ---- Step 1: Ingestion -------------------------------------------------------
step_ingest = ProcessingStep(
    name="CreditIngest",
    processor=processor,
    inputs=[ProcessingInput(source=BASE, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(output_name="ingested",
                              source="/opt/ml/processing/ingested",
                              destination=f"{OUT}/ingested")],
    code="data_ingestion.py",
)

# ---- Step 2: Preprocessing ---------------------------------------------------
step_preprocess = ProcessingStep(
    name="CreditPreprocess",
    processor=processor,
    inputs=[ProcessingInput(
        source=step_ingest.properties.ProcessingOutputConfig.Outputs["ingested"].S3Output.S3Uri,
        destination="/opt/ml/processing/ingested")],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train",
                         destination=f"{OUT}/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test",
                         destination=f"{OUT}/test"),
    ],
    code="preprocessing.py",
)

# ---- Step 3: Training --------------------------------------------------------
step_train = TrainingStep(
    name="CreditTrain",
    estimator=estimator,
    inputs={"train": step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri},
)

# ---- Step 4: Evaluation ------------------------------------------------------
eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation",
                           path="evaluation.json")
step_eval = ProcessingStep(
    name="CreditEval",
    processor=processor,
    inputs=[
        ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                        destination="/opt/ml/processing/model"),
        ProcessingInput(
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"),
    ],
    outputs=[ProcessingOutput(output_name="evaluation",
                              source="/opt/ml/processing/evaluation",
                              destination=f"{OUT}/eval")],
    code="evaluation.py",
    property_files=[eval_report],
)

# ---- Rangkai & jalankan ------------------------------------------------------
pipeline = Pipeline(
    name="Credit-Local-Workflow",
    steps=[step_ingest, step_preprocess, step_train, step_eval],
    sagemaker_session=local_session,
)

if __name__ == "__main__":
    import json

    print("Registrasi & menjalankan pipeline (local mode)...")
    pipeline.upsert(role_arn=role)
    pipeline.start()   # local mode: blocking; log tiap step tampil inline di atas

    # Konfirmasi hasil dari file evaluasi (kalau semua step sukses).
    report_path = os.path.join(BASE, "eval", "evaluation.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            metrics = json.load(f)["multiclass_classification_metrics"]
        print("\n=== PIPELINE SELESAI ===")
        print(f"Macro F1 = {metrics['macro_f1']['value']:.4f} | "
              f"Accuracy = {metrics['accuracy']['value']:.4f}")
        print(f"(detail: {report_path})")
    else:
        print("\nPipeline selesai, tapi evaluation.json belum ada — "
              "cek log tiap step di atas untuk melihat error.")
