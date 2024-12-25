import subprocess
import re
import ast
import os
os.environ['TERM'] = 'dumb'
class TreeNode:
    def __init__(self, name, node_type):
        self.name = name
        self.node_type = node_type  # "Class" or "Function"
        self.children = []
        self.relationships = []

    def add_child(self, child_node, relationship_type):
        if child_node not in self.children:
            self.children.append(child_node)
            self.relationships.append(relationship_type)

    def __repr__(self):
        return f"TreeNode(name={self.name}, type={self.node_type}, relationships={self.relationships})"


def run_joern_query(query):
    process = subprocess.Popen(
        ['joern'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(query.encode())
    if stderr:
        print("Error:", stderr.decode())
    return stdout.decode()

def clean_string(input_str):
    sections = re.split(r'joern>', input_str)
    if len(sections) < 3:
        return "Insufficient 'joern>' tags found"
    clean_section = re.sub(r'\x1b\[[0-9;]*m', '', sections[-2]).strip()
    return clean_section


def get_class_definitions(import_project, project_name):
    query = f'''
    importCode(inputPath="{import_project}", projectName="{project_name}")
    cpg.typeDecl.nameNot("<global>").name.l
    '''
    result = run_joern_query(query)
    return [line.strip() for line in result.splitlines()]


def get_inheritance_relations(class_name):
    query = f'cpg.typeDecl.name("{class_name}").inheritsFromTypeFullName.l'
    result = run_joern_query(query)
    return [line.strip() for line in result.splitlines()]


def get_methods(class_name):
    query = f'cpg.typeDecl.name("{class_name}").method.name.l'
    result = run_joern_query(query)
    return [line.strip() for line in result.splitlines()]


def add_function_call_relations(method_name):
    query = f'cpg.method.name("{method_name}").callOut.name.l'
    result = run_joern_query(query)
    return [line.strip() for line in result.splitlines()]


def build_relation_tree(project_name, import_project):
    root = TreeNode(name=project_name, node_type="Project")

    class_definitions = get_class_definitions(import_project, project_name)
    class_nodes = {}
    for class_name in class_definitions:
        class_node = TreeNode(name=class_name, node_type="Class")
        class_nodes[class_name] = class_node
        root.add_child(class_node, "Inheritance")

    for class_name, class_node in class_nodes.items():
        inheritance_relations = get_inheritance_relations(class_name)
        for parent_class_name in inheritance_relations:
            if parent_class_name in class_nodes:
                parent_class_node = class_nodes[parent_class_name]
                parent_class_node.add_child(class_node, "Inheritance")

    for class_name, class_node in class_nodes.items():
        method_definitions = get_methods(class_name)
        for method_name in method_definitions:
            method_node = TreeNode(name=method_name, node_type="Function")
            class_node.add_child(method_node, "Encapsulation")

            called_functions = add_function_call_relations(method_name)
            for func_name in called_functions:
                func_node = TreeNode(name=func_name, node_type="Function")
                method_node.add_child(func_node, "Call")
    
    return root


def display_tree(node, depth=0, visited=None):
    if visited is None:
        visited = set()
    if node in visited:
        print("  " * depth + f"{node.name} ({node.node_type}) [Already displayed]")
        return
    
    visited.add(node)
    
    print("  " * depth + f"{node.name} ({node.node_type})")
    for i, child in enumerate(node.children):
        relationship = node.relationships[i]
        print("  " * (depth + 1) + f"└─[{relationship}]─> ", end="")
        display_tree(child, depth + 2, visited)



def main():
    import_project = ""
    project_name = ""
    relation_tree = build_relation_tree(project_name, import_project)
    print("\nThe Tree are:")
    display_tree(relation_tree)

if __name__ == "__main__":
    main()