import os

def clear_txt_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'w') as file:
                file.write('') 
            print(f"Empty file: {file_path}")

folder_path = '' 
clear_txt_files_in_folder(folder_path)