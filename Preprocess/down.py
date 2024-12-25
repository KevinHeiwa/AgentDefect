import pandas as pd
import os
import subprocess

# Load the CSV file
file_path = '/home/ningkw/Agentable/Experment_Agent/RQ1_Agent.csv'
df = pd.read_csv(file_path, header=None)

# Extract the second column which contains the git repository URLs
repo_urls = df[1].tolist()

# Define the base directory where projects will be cloned
base_clone_directory = '/home/ningkw/Agentable/Experment_Agent/Agent'

# Create the base directory if it doesn't exist
os.makedirs(base_clone_directory, exist_ok=True)

# Loop through each URL and clone the repositories into individual subdirectories
for url in repo_urls:
    try:
        # Extract the project name from the repository URL
        project_name = url.split('/')[-1].replace('.git', '').split('?')[0]
        project_dir = os.path.join(base_clone_directory, project_name)
        
        # Only clone if the directory doesn't already exist
        if not os.path.exists(project_dir):
            # Prepare the git clone command
            command = ['git', 'clone', url, project_dir]
            # Execute the git clone command
            subprocess.run(command, check=True)
            print(f"Cloned: {url} into {project_dir}")
        else:
            print(f"Project {project_name} already exists. Skipping...")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {url}: {e}")

