import os, shutil
import kagglehub

dataset = "carlthome/gtzan-genre-collection"
print("Downloading:", dataset)
path = kagglehub.dataset_download(dataset)
print("Downloaded to:", path)

target = os.path.join("music_data", "gtzan", "genres")
os.makedirs(target, exist_ok=True)
shutil.copytree(path, target, dirs_exist_ok=True)
print("Copied into:", target)
