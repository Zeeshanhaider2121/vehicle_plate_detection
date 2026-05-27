import os

# Set env vars before the app module is imported so Settings reads them.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_plateflow.db")
os.environ.setdefault("MOCK_INFERENCE_IF_UNAVAILABLE", "true")
os.environ.setdefault("COLAB_INFER_URL", "")
os.environ.setdefault("MULTI_CAMERA_GATE_MODE", "false")
os.environ.setdefault("MULTI_CAMERA_CAMERA_ROLES", "")
