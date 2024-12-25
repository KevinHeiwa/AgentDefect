
import os
import shutil

def copy_folder(src_folder, dest_folder1, dest_folder2):
    shutil.copytree(src_folder, dest_folder1, dirs_exist_ok=True)
    shutil.copytree(src_folder, dest_folder2, dirs_exist_ok=True)



def remove_non_program_files(folder):
    valid_extensions = ['.py', '.java', '.cpp', '.c', '.js', '.html', '.css', '.php', '.go', '.rb'] 
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            if not any(file.endswith(ext) for ext in valid_extensions):
                os.remove(file_path)
                print(f"Deleted non-program file: {file_path}")


def remove_non_python_files(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.endswith('.py'):
                os.remove(file_path)
                print(f"Deleted non-Python file: {file_path}")

def main():
    src_folder = ""     
    dest_folder1 = ""  
    dest_folder2 = ""  
    copy_folder(src_folder, dest_folder1, dest_folder2)
 
    # print(f"Cleaning up non-program files in {dest_folder1}")
    # remove_non_program_files(dest_folder1)
    print(f"Cleaning up non-Python files in {dest_folder2}")
    remove_non_python_files(dest_folder2)

if __name__ == "__main__":
    main()