from utils.tools import get_name
import os
import shutil


root = "/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e/wbwwbw"
root_new = "/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e/pp"
os.makedirs(root_new, exist_ok=True)
folder_name = get_name(root)
print(folder_name)
for folder in folder_name:
    save_root = os.path.join(root_new, folder)
    os.makedirs(save_root, exist_ok=True)
    folder2_name = get_name(os.path.join(root, folder))
    for folder2 in folder2_name:
        file_name = get_name(os.path.join(root, folder, folder2), mode_folder=False)
        for i in range(len(file_name)):
            nn = file_name[i].split(".")[-1]
            shutil.copy(os.path.join(root, folder, folder2, file_name[i]), os.path.join(save_root, str(i) + folder2 + "." + nn))

