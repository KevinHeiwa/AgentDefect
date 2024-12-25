import os
import shutil

def split_subfolders_into_four(target_folder):
    if not os.path.exists(target_folder):
        print("Warning!")
        return

    subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]
    total_subfolders = len(subfolders)
    print(f"Number of file: {total_subfolders}")
    
    if total_subfolders == 0:
        print("Warning!")
        return
    chunk_size = (total_subfolders + 3) // 4 
    chunks = [subfolders[i:i + chunk_size] for i in range(0, total_subfolders, chunk_size)]
    
    for i, chunk in enumerate(chunks, start=1):
        new_folder = os.path.join(target_folder, f"Group_{i}")
        os.makedirs(new_folder, exist_ok=True)
        print(f"New file: {new_folder}")
        
        for subfolder in chunk:
            src_path = os.path.join(target_folder, subfolder)
            dest_path = os.path.join(new_folder, subfolder)
            shutil.move(src_path, dest_path)
            print(f"From {subfolder} to {new_folder}")
    
    print("")

target_folder = ""  
split_subfolders_into_four(target_folder)