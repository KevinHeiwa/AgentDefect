import re

def count_unique_projects(file_path):
    projects = set()
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"([^/]+)/", line)
            if match:
                projects.add(match.group(1))  
    return len(projects), projects

# 示例调用
file_path = "" 
unique_project_count, unique_projects = count_unique_projects(file_path)

print(f"Total of {unique_project_count} Projects")
print("\n".join(unique_projects))