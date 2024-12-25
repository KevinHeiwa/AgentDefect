
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

only_program_folder = ""

detect_script = ""

def process_project(project, pbar):
    project_path = os.path.join(only_program_folder, project)
    
    if os.path.isdir(project_path):
        pbar.set_description(f"Processing {project}...")
        subprocess.run(['python3', detect_script, project_path, project])
        pbar.update(1)  
        return f"{project} processed."


def main():
    projects = [project for project in os.listdir(only_program_folder) if os.path.isdir(os.path.join(only_program_folder, project))]
    total_projects = len(projects)

    with ThreadPoolExecutor(max_workers=12) as executor: 
        with tqdm(total=total_projects, desc="Processing projects", unit="project") as pbar:
            futures = {executor.submit(process_project, project, pbar): project for project in projects}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(result)  
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")

if __name__ == "__main__":
    main()