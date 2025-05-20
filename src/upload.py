from huggingface_hub import HfApi
import os

api = HfApi(token = os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path = "/home/zyy/LLaMA-Factory-pre/models/QCompiler-Llama3.2-3B",
    repo_id = "KeriaZhang/QCompiler-Llama3.2-3B",
    repo_type = "model",
)
