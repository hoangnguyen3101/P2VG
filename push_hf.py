from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="/home/hoangnv/AICD_HA/SPINE_BASE/P2VG/output_gemma3_fold5", repo_id="hoangnguyen311111/p2vg_fold5", repo_type="model")
