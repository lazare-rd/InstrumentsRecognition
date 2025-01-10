from huggingface_hub import login
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.getenv('HF_TOKEN')
login(TOKEN)

api = HfApi()

api.delete_folder(
    path_in_repo="wav",
    repo_id="lazarerd/ClassicInstruments",
    repo_type="dataset"
)

api.upload_large_folder(
    folder_path="data",
    repo_id="lazarerd/ClassicInstruments",
    repo_type="dataset"
)


