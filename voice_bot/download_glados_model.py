"""Download GLaDOS model from HuggingFace."""
import os
import sys

sys.path.insert(0, r"C:\Users\trist\Downloads\muster-roll-extracted\voice_bot\venv\Lib\site-packages")

from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "WarriorMama777/GLaDOS_TTS"
subfolder = "Models/Style-Bert_VITS2/Portal_GLaDOS_v1"
local_dir = r"C:\Users\trist\Downloads\muster-roll-extracted\voice_bot\model_assets/Portal_GLaDOS_v1"

os.makedirs(local_dir, exist_ok=True)

# List all repo files (no subdirectory param - use the older API)
print("Listing all repo files...")
all_files = list_repo_files(repo_id, repo_type="model")
print("Total files:", len(all_files))
print("Files in Style-Bert_VITS2:", [f for f in all_files if 'Style-Bert_VITS2' in f])

# Filter for our model's files
target_files = [f for f in all_files if f.startswith(subfolder) and f.endswith(('.safetensors', '.json', '.npy'))]
print("\nTarget files:", target_files)

for f in target_files:
    filename = f.split('/')[-1]
    # The path in the repo includes the subfolder prefix
    dest = hf_hub_download(
        repo_id=repo_id,
        filename=f,
        local_dir=local_dir,
        force_download=False,
    )
    fsize = os.path.getsize(dest)
    print(f"Downloaded: {filename} ({fsize} bytes)")

print("\nLocal files now:")
for item in os.listdir(local_dir):
    path = os.path.join(local_dir, item)
    size = os.path.getsize(path)
    print(f"  {item} ({size} bytes)")
