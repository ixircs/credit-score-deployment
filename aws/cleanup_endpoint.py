# cleanup_endpoint.py — Hapus SageMaker endpoint + config + model (hemat biaya).
# =============================================================================
# Jalankan setelah selesai screenshot / demo agar tidak dikenai biaya per jam
# dan agar deploy ulang tidak bentrok nama.
# python cleanup_endpoint.py
import boto3

ENDPOINT_NAME = "credit-score-endpoint"
REGION = "us-east-1"


def _try(desc, fn):
    print(desc)
    try:
        fn()
        print("  OK")
    except Exception as e:
        print(f"  (lewati) {e}")


def main():
    sm = boto3.client("sagemaker", region_name=REGION)
    _try(f"Hapus endpoint: {ENDPOINT_NAME}",
         lambda: sm.delete_endpoint(EndpointName=ENDPOINT_NAME))
    _try(f"Hapus endpoint-config: {ENDPOINT_NAME}",
         lambda: sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME))
    print("Cleanup selesai.")


if __name__ == "__main__":
    main()
