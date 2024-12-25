import subprocess
import os
import re
import json
import ast
from openai import OpenAI # type: ignore
from collections import deque
import sys


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="",
    base_url=""
)


Model = ""
result_file = ""

import time
from functools import wraps

# Decorator to measure execution time of functions
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start timer
        result = func(*args, **kwargs)   # Execute the function
        end_time = time.perf_counter()  # End timer
        execution_time = end_time - start_time
        # print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper


# Function to execute Joern queries and extract class information
 
def get_class_info_from_joern(importProject, projectName):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.typeDecl.filter(_.inheritsFromTypeFullName.exists(_.matches(".*BaseModel.*"))).l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    ## print(result_clean)
    return result_clean


def clean_string(input_str):
    sections = re.split(r'joern>', input_str)
    if len(sections) < 3:
        return "Error"
    target_section = sections[-2]
    clean_section = re.sub(r'\x1b\[[0-9;]*m', '', target_section)
    clean_section = clean_section.strip()
    return clean_section



def extract_class_info(input_string):
    class_info_pattern = r'filename = "(.*?)",\n\s+fullName = "(.*?):<module>\.(.*?)",\n\s+.*?lineNumber = Some\(value = (\d+)\)'
    matches = re.findall(class_info_pattern, input_string, re.MULTILINE | re.DOTALL)
    class_info_list = []
    for match in matches:
        class_info = {
            'filename': match[0],
            'full_file_name': match[1],  
            'classname': match[2],
            'line_number': match[3]
        }
        class_info_list.append(class_info)

    return class_info_list


 
def extract_class_code_1(folder_path, class_info_list):
    class_code_dict = {}  
    output_data = []  

    for class_info in class_info_list:
        file_path = os.path.join(folder_path, class_info['filename'])          
        if not os.path.exists(file_path):
            output_data.append(f"File {file_path} not found.\n")
            continue
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()

        class_name = class_info['classname']
        class_start = re.search(rf'class {class_name}\b', source_code)
        if not class_start:
            output_data.append(f"Class {class_name} not found in {file_path}.\n")
            continue

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            output_data.append(f"Error parsing {file_path}: {e}\n")
            continue

        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break
        if not class_node:
            output_data.append(f"Class {class_name} not found in {file_path}.\n")
            continue

        class_code_lines = source_code.splitlines()[class_node.lineno - 1:class_node.end_lineno]
        class_code = "\n".join(class_code_lines)
        if class_name not in class_code_dict:
            class_code_dict[class_name] = {}
        class_code_dict[class_name][file_path] = class_code
        combined_string = f"Class {class_name} from {file_path}:\n{class_code}\n\classend\n"
        output_data.append(combined_string)

    return class_code_dict, output_data



def prompt(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """Please analyze the following code and identify whether each class qualifies as an “Agent initialization class” or “LLM initialization class” based on the following criteria: 1.Initialization Class of Agent: A class is considered an Agent initialization class if it explicitly contains code that directly sets up or initializes an Agent instance, configuring its core functions, attributes, or behaviors as part of the Agent framework. Please avoid marking classes as Agent initialization classes if they only have peripheral relationships with the Agent. 2.Initialization Class of LLM: A class is considered an LLM initialization class only if it contains essential code that directly configures or initializes an LLM instance, including specifying model parameters or invoking key LLM setup functions. Classes should not be marked as LLM initialization classes based on incidental or minor references to LLM-related elements. Please output only the result in the following JSON format, without additional commentary: { "Class Name": "Please print the name of the Class.", "Initialization class of Agent": "yes or no?", "Initialization class of LLM": "yes or no?" }\n """
    label = """[Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + analysis_object
    return combined_string


def llm_check(content):
    new_content = prompt(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results

 
def process_content(content):
    sections = content.split('\n\\classend\n')
    sections = [section for section in sections if section.strip()]
    chunks = [sections[i:i + 10] for i in range(0, len(sections), 10)]
    final_results = []
    for chunk in chunks:
        the_new_content = ''.join(chunk) + '\n\n'
        the_result = llm_check(the_new_content)
        final_results.append(the_result)
    return ''.join(final_results)


def extract_relevant_json(data):
    pattern = re.compile(r'{[^}]+}')
    matches = pattern.findall(data)
    extracted_json_list = []
    for match in matches:
        try:
            json_obj = json.loads(match)         
            if all(key in json_obj for key in ["Class Name", "Initialization class of Agent", "Initialization class of LLM"]):
                extracted_json_list.append(json.dumps(json_obj, indent=2))
        except json.JSONDecodeError:
            continue
    return '\n'.join(extracted_json_list)



def extract_class_attributes_from_dict(class_dict):
    extracted_attributes = {}
    for class_name, file_info in class_dict.items():
        extracted_attributes[class_name] = {}
        for file_path, attributes in file_info.items():
            extracted_attributes[class_name][file_path] = []
            for attr_name, attr_type, init_value in attributes:
                extracted_attributes[class_name][file_path].append({
                    'name': attr_name,
                    'type': attr_type,
                    'initial_value': init_value
                })

    return extracted_attributes




def read_file_to_string(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        return file_content
    except FileNotFoundError:
        return f"File {file_path} not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"
    


def parse_classes(input_string):
    class_pattern = re.compile(r'"Class Name":\s*"([^"]+)"')
    llm_init_pattern = re.compile(r'"Initialization class of LLM":\s*"([^"]+)"')
    agent_init_pattern = re.compile(r'"Initialization class of Agent":\s*"([^"]+)"')

    class_names = class_pattern.findall(input_string)
    llm_inits = llm_init_pattern.findall(input_string)
    agent_inits = agent_init_pattern.findall(input_string)

    llm_classes = []
    agent_classes = []
    all_classes = class_names  
    for i in range(len(class_names)):
        if llm_inits[i] == "yes":
            llm_classes.append(class_names[i])
        if agent_inits[i] == "yes":
            agent_classes.append(class_names[i])
    if agent_classes:
        return {"Agent Classes": agent_classes}
    else:
        return {"All Classes": all_classes}



def get_file_paths_by_class_name(class_code_dict, class_name):
    if class_name in class_code_dict:
        file_paths = class_code_dict[class_name].keys()
        return list(file_paths)
    else:
        return f"Class '{class_name}' not exit"



def extract_agent_info(text):
    json_pattern = r'\{[^\}]*\}'
    matches = re.findall(json_pattern, text)

    extracted_data = []
    for match in matches:
        try:
            json_data = json.loads(match)
            extracted_data.append(json_data)
        except json.JSONDecodeError:
            continue 
    return extracted_data



def extract_agent_classes(result_classes):
    if 'Agent Classes' in result_classes:
        agent_classes = result_classes['Agent Classes']
        return agent_classes
    else:
        return []


 
def handle_agent_classes_logic(result_classes, class_code_dict, folder_path, output_file,importProject, projectName):
    # print(f"Agent class is {result_classes}")
    ## print(f"{class_code_dict}")
    agent_classes = extract_agent_classes(result_classes)
    ## print(f"{agent_classes}")
    return agent_classes


def extract_user_defined_functions(project_path):
    user_defined_functions = []
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in {'venv', 'site-packages', '__pycache__'}]

        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    func_defs = re.findall(r'\bdef\s+(\w+)\s*\(', file_content)
                    user_defined_functions.extend([f"{func_name}" for func_name in func_defs])
    user_defined_functions = list(dict.fromkeys(user_defined_functions))

    return user_defined_functions



def extract_class_methods(classes_dict):
    functions = []
    
    for class_name, files in classes_dict.items():
        for file_path, class_code in files.items():
            # Find all function definitions within the class code
            func_defs = re.findall(r'\bdef\s+(\w+)\s*\(', class_code)
            # Append functions along with class name to specify their origin
            functions.extend([f"{func_name}" for func_name in func_defs])
    
    return functions


def extract_specific_functions_from_project(project_path, function_names):
    """Extract specified functions from Python files in a project directory and save them to file and a dictionary."""
    extracted_functions = []  # List to store all extracted function information


    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions = extract_specific_functions_from_file(file_path, function_names)
                if functions:
                    for func in functions:
                        combined_code = f"File path: {file_path}\n{func['code']}"
                        extracted_functions.append({
                            "name": func["name"],
                            "code": combined_code
                        })

    return extracted_functions


def extract_specific_functions_from_file(file_path, function_names):
    """Extract specific functions by name from a Python file."""
    with open(file_path, "r", encoding="utf-8") as file:
        source_code = file.read()
    tree = ast.parse(source_code)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            # Extract function name
            func_name = node.name
            # Extract function source code
            func_source = ast.get_source_segment(source_code, node)
            # Append the function information as a dictionary entry
            functions.append({
                "name": func_name,
                "path": file_path,
                "code": func_source
            })

    return functions


def group_functions_in_batches(extracted_data, batch_size=10):
    """
    Group functions from extracted_data into batches of specified size.
    Each batch is stored as a single string containing the code of batch_size functions.
    """
    grouped_functions = []  # List to store all batches of function code
    current_batch = []      # Temporary list to store current batch of function code

    for i, func_info in enumerate(extracted_data):
        # Add function code to the current batch
        current_batch.append(func_info["code"])

        # If current batch reaches the batch size, join into a single string and reset
        if (i + 1) % batch_size == 0:
            grouped_functions.append("\n".join(current_batch))
            current_batch = []

    # Handle remaining functions if they do not fill up the last batch
    if current_batch:
        grouped_functions.append("\n".join(current_batch))

    return grouped_functions


def prompt_check_method(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """Please analyze the following code and identify whether each method qualifies as an “Agent initialization method” or “LLM initialization method” based on the following criteria: 1.Initialization method of Agent: A method is considered an Agent initialization method if it explicitly contains code that directly sets up or initializes an Agent instance, configuring its core functions, attributes, or behaviors as part of the Agent framework. Please avoid marking method as Agent initialization method if they only have peripheral relationships with the Agent. 2.Initialization method of LLM: A method is considered an LLM initialization method only if it contains essential code that directly configures or initializes an LLM instance, including specifying model parameters or invoking key LLM setup functions. Method should not be marked as LLM initialization method based on incidental or minor references to LLM-related elements. Please output only the result in the following JSON format, without additional commentary: { "method Name": "Please print the name of the method.", "Initialization method of Agent": "yes or no?", "Initialization method of LLM": "yes or no?" }\n """
    label = """[Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + analysis_object
    return combined_string


def llm_check_method(content):
    new_content = prompt_check_method(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results

def extract_relevant_json_1(data):
    pattern = re.compile(r'{[^}]+}')
    matches = pattern.findall(data)
    extracted_json_list = []
    for match in matches:
        try:
            json_obj = json.loads(match)         
            if all(key in json_obj for key in ["method Name", "Initialization method of Agent", "Initialization method of LLM"]):
                extracted_json_list.append(json.dumps(json_obj, indent=2))
        except json.JSONDecodeError:
            continue
    return '\n'.join(extracted_json_list)


def extract_relevant_json_3(data):
    pattern = re.compile(r'{[^}]+}')
    matches = pattern.findall(data)
    extracted_json_list = []
    for match in matches:
        try:
            json_obj = json.loads(match)         
            if all(key in json_obj for key in ["method Name", "is Tool Implementation Instance"]):
                extracted_json_list.append(json.dumps(json_obj, indent=2))
        except json.JSONDecodeError:
            continue
    return '\n'.join(extracted_json_list)


def parse_methods(input_string):
    method_pattern = re.compile(r'"method Name":\s*"([^"]+)"')
    llm_init_pattern = re.compile(r'"Initialization method of LLM":\s*"([^"]+)"')
    agent_init_pattern = re.compile(r'"Initialization method of Agent":\s*"([^"]+)"')

    method_names = method_pattern.findall(input_string)
    llm_inits = llm_init_pattern.findall(input_string)
    agent_inits = agent_init_pattern.findall(input_string)

    llm_method = []
    agent_method = []
    all_classes = method_names  
    for i in range(len(method_names)):
        if llm_inits[i] == "yes":
            llm_method.append(method_names[i])
        if agent_inits[i] == "yes":
            agent_method.append(method_names[i])

    if agent_method:
        return {"Agent Method": agent_method}
    else:
        return {"No Method": None}
    

def prompt_for_check_method_attri(content):
    system_message = """You are a developer of large language model applications, capable of accurately identifying elements in the code and clearly distinguishing the attributes of LLMs based on given criteria. \n """
    analysis_message = """The following code is related to the initialization of an LLM or Agent. Please carefully read the code and perform the following analysis: First, analyze whether the code includes any LLM initialization parameters. If so, extract the values of these initialization parameters and make the following determinations: Then, based on existing knowledge and factors such as the model’s name, assess whether the model is a conversational model. For example, well-known models like the GPT series or Llama series, or models whose names contain keywords like “Chat,” are typically conversational models. Furthermore, based on existing knowledge and the model’s name, determine whether the model has sufficient generative capabilities. Well-known models, such as the GPT series or Llama series, are generally considered to have adequate generative capabilities. It is important to note that if the model is designed for a specific task, such as code-related tasks (often containing the keyword “code” in the name), these models may not be considered to have sufficient generative capabilities.Finally, only return the result must in the following JSON format: { "LLM Name": "The name of the model (if none or cannot be extracted, None)", "Is the model a Chat model?": "yes or no?", "Does the model have sufficient generative capabilities?": "yes or no?" }\n """
    label = """[Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string

def llm_check_method_attri(content):
    new_content = prompt_for_check_method_attri(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=, 
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    result = extract_agent_info(results)
    return result



def handle_no_classes_logic(result_classes, folder_path, output_file_1, class_code_dict,importProject, projectName):
    all_method = extract_user_defined_functions(importProject)
    method_in_class = extract_class_methods(class_code_dict)
    invidial_method = remove_elements(all_method,method_in_class)
    order_invidial_method = reorder_class_list(invidial_method)
    extracted_data = extract_specific_functions_from_project(importProject,order_invidial_method)
    grouped_functions = group_functions_in_batches(extracted_data)
    method_check = ""
    for i in grouped_functions:
        method_check_1 = llm_check_method(i)
        method_check += method_check_1
        method_check += '\n'
    No_res = extract_relevant_json_1(method_check)
    res_method = parse_methods(No_res)
    result_dict = {} 
    if "Agent Method" in res_method:
        function_names = res_method.get("Agent Method", [])
        for func_name in function_names:
            for func_data in extracted_data:
                if func_data["name"] == func_name:
                    path = func_data["path"]
                    code = func_data["code"]
                    if func_name not in result_dict:
                        result_dict[func_name] = {} 
                    result_dict[func_name][path] = code
 
                else:
                    print("Jump Function")
        return result_dict, func_name
    else:
        return None,None
    



def extract_classes_by_name(class_dict, agent_class):
    if class_dict is None:
        # # print("class_dict is None")
        return {}

    if not isinstance(agent_class, (list, tuple, set)):
        # # print(f"agent_class is None or TypeError: {type(agent_class)}")
        return {}

    extracted_classes = {}

    for class_name in agent_class:
        if class_name in class_dict:
            extracted_classes[class_name] = class_dict[class_name]
        else:
            print(f"class {class_name} is not found.")
    
    return extracted_classes

 
def get_class_info_from_joern_tool():
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.typeDecl.name.l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean

 
def get_method_info_from_joern():
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.method.name.l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean


def clean_string(input_str):
    sections = re.split(r'joern>', input_str)
    if len(sections) < 3:
        return "Error"
    target_section = sections[-2]
    clean_section = re.sub(r'\x1b\[[0-9;]*m', '', target_section)
    clean_section = clean_section.strip()
    return clean_section


def parse_2_string(res1_str):
    res1_list = re.findall(r'"(.*?)"|\'(.*?)\'', res1_str)
    res1 = [item for sublist in res1_list for item in sublist if item]
    
    return res1


def extract_valid_class_names(res1_str):

    res1 = parse_2_string(res1_str)

    invalid_keywords = ["<", ">", "<module>", "<meta>", "<body>", "<metaClassCallHandler>", "<fakeNew>", "<metaClassAdapter>","ANY"]
    valid_class_names = set()
    
    for item in res1:
        if not any(keyword in item for keyword in invalid_keywords):
            valid_class_names.add(item)
    
    return list(valid_class_names)


def remove_elements(list1, list2):
    #return [item for item in list1 if item not in list2]
    return [item for item in list1 if item not in list2]


 
def get_method_info_from_joern():
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.method.name.l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean



 
def query_class_inheritance(classes):

    inheritance_dict = {}

    for class_name in classes:
        joern_query = f"""
        importCode(inputPath="{importProject}", projectName="{projectName}")
        cpg.typeDecl.name("{class_name}").inheritsFromTypeFullName.l
        """
        
        try:
            process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(joern_query.encode())
            result = stdout.decode('utf-8')
            result_clear = clean_string(result)
            result_clean = clean_inheritance_result(result_clear)
            if 'List()' in result_clean or 'val res' in result_clean:
                inheritance_dict[class_name] = None
            else:
                inheritance_dict[class_name] = result_clean if result_clean else None
        
        except Exception as e:
            inheritance_dict[class_name] = f"Error: {str(e)}"
    
    return inheritance_dict


 
def clean_inheritance_result(result):
    match = re.findall(r'List\("([^"]+)"\)', result)
    if match:
        clean_results = []
        for item in match:
           
            clean_name = item.split(".")[-1]  
            clean_results.append(clean_name)
        return ', '.join(clean_results)  
    return result


 
def build_inheritance_tree(inheritance_dict):
    tree = {}

    def add_to_tree(class_name):
        if class_name not in tree:
            tree[class_name] = []
        for subclass, superclass in inheritance_dict.items():
            if superclass == class_name:
                tree[class_name].append(subclass)
                add_to_tree(subclass)  
    for class_name in inheritance_dict.keys():
        add_to_tree(class_name)
    
    to_remove = [cls for cls in tree if not tree[cls] and cls not in inheritance_dict.values()]
    for cls in to_remove:
        del tree[cls]
    
    return tree

 
def level_order_traversal(tree):
    queue = deque()
    root_nodes = [node for node in tree.keys() if all(node not in v for v in tree.values())]
    visited = set()
    for root in root_nodes:
        if root not in visited:
            queue.append(root)
            visited.add(root)
    level = []
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            if node in tree:
                for child in tree[node]:
                    if child not in visited:
                        queue.append(child)
                        visited.add(child)  
        tool_related = [cls for cls in current_level if "Tool" in cls or "tool" in cls]
        non_tool_related = [cls for cls in current_level if "Tool" not in cls and "tool" not in cls]
        current_level = tool_related + non_tool_related
        level.append(current_level)
    return level


 
def extract_class_code(filename, class_name, start_line):
    class_code = []
    file_in = os.path.join(importProject, filename)
    try:
        with open(file_in, 'r') as file:
            lines = file.readlines()

        inside_class = False
        indent_level = None
        
        for i, line in enumerate(lines[start_line - 1:], start=start_line):
            if line.strip().startswith(f"class {class_name}") and not inside_class:
                inside_class = True
                indent_level = len(line) - len(line.lstrip()) 
                class_code.append(line)
            elif inside_class:
                current_indent = len(line) - len(line.lstrip())
                
                if current_indent > indent_level or line.strip() == '':
                    class_code.append(line)
                else:
                    break
        return ''.join(class_code) if class_code else None
    except FileNotFoundError:
        # # print(f"Error: The file '{file_in}' was not found.")
        return None
    except Exception as e:
        # # print(f"An error occurred: {e}")
        return None
    
 
def query_class_details(class_name):
    
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.typeDecl.name("{class_name}").l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    results = extract_filenames_and_lines(result_clean)
    return results

    

 
def extract_filenames_and_lines(input_string):
    # Regex pattern to capture TypeDecl elements' filename and lineNumber fields
    filename_pattern = r'filename = "([^"]+)"'
    line_number_pattern = r'lineNumber = Some\(value = (\d+)\)'

    # Extract all filenames and line numbers from the input string
    filenames = re.findall(filename_pattern, input_string)
    line_numbers = re.findall(line_number_pattern, input_string)

    # Check if all filenames are "<unknown>"
    if all(filename == "<unknown>" for filename in filenames):
        return None

    # Extract and return non-unknown filename and corresponding line number pairs
    result = []
    for filename, line_number in zip(filenames, line_numbers):
        if filename != "<unknown>":
            result.append((filename, int(line_number) if line_number != 'None' else None))

    return result



def llm_check_tool_initialization_prompt(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """After [Class Code], the code for a class is provided. We are looking to find the initial class of the Tool in an Agent project, which other tool instances inherit when implemented. First, carefully read the provided code, then determine whether this code is related to Tool, and further assess whether it is the initial class of Tool. Return the results in the following JSON format: {"Class Name": "Please print the name of the Class.", "Is it related to Tool?": "yes or no?", "Is it the initial class of Tool?": "yes or no?"
} \n """
    label = """[Class Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string



 
def llm_check_tool_initialization_class(content):
    new_content = llm_check_tool_initialization_prompt(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results



def extract_tool_class_info(str_input):
    match = re.search(r'{[\s\S]*}', str_input)  
    if not match:
        # print("No JSON-like structure found in the input string.")
        return None
    json_str = match.group()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # print("Failed to parse JSON string.")
        return None
    is_related_to_tool = data.get("Is it related to Tool?")
    is_initial_class_of_tool = data.get("Is it the initial class of Tool?")
    if is_related_to_tool == "yes" and is_initial_class_of_tool == "yes":
        return data.get("Class Name")
    return None


 
def process_layers(layers):
    for level in layers:
        # print(f"Processing level: {level}")
        for class_name in level:
            details  = query_class_details(class_name)
            
            if details:
                filename, line_number = details[0]                
               
                class_code = extract_class_code(filename, class_name, line_number)               
                llm_check_res = llm_check_tool_initialization_class(class_code)
                # # print(f"the res is:{llm_check_res}")
                find_res = extract_tool_class_info(llm_check_res)
                if find_res is not None:
                    return class_name, class_code
    return None,None



def llm_check_tool_use_prompt(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """After [Class Code], there is code for a tool class, which is inherited by each specific tool implementation. This class contains many functions, and we aim to find one function that is meant for external calls—the one that actually runs the tool. When the project calls the tool, it will directly invoke this function from the tool. So, first, carefully read the class code and the functions it contains. Next, identify the function that meets these criteria. Finally, only return the results in the following JSON format: {"Class Name": "Please print the name of the Class.","Name of the function that meets the criteria": "Please output the function name (if none, return None)"}"""
    label = """[Class Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string

 
def llm_check_tool_use_class(content):
    new_content = llm_check_tool_use_prompt(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=0.6,)
    results = response.choices[0].message.content
    return results

 
def extract_tool_def_info(str_input):
    match = re.search(r'{[\s\S]*}', str_input) 
    if not match:
        print("No JSON-like structure found in the input string.")
        return []
    
    json_str = match.group()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # print("Failed to parse JSON string.")
        return []
    
    class_name = data.get("Class Name")
    use_def = data.get("Name of the function that meets the criteria")

    if isinstance(use_def, list):
        return [item for item in use_def if item and item != "None"]

    if isinstance(use_def, str) and use_def != "None":
        return [use_def]

    return []

 
def analyze_class_code(class_code, defname):
    tree = ast.parse(class_code)
    called_functions = []
    class FunctionCallVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Call):
                    if isinstance(stmt.func, ast.Attribute) and stmt.func.attr == defname:
                        called_functions.append({
                            'function_name': node.name,
                            'function_code': ast.unparse(node)  
                        })
            self.generic_visit(node)
    visitor = FunctionCallVisitor()
    visitor.visit(tree)
    
    return called_functions


 
def extract_function_calls(class_dict, defname):
    if defname:
        defname = list(set(defname))

    with open(result_file, 'a') as f:
        results = {}
        check_res_2 = None
        if not class_dict or class_dict == {} or class_dict is None:
            f.write(f"Not find Agent class: {class_dict}. \n")
        if not defname or defname == '' or defname is None or defname == []:
            f.write(f"Not find LLM use function: {defname}. \n")
            
        for class_name, file_info in class_dict.items():
            for file_path, class_code in file_info.items():
                f.write(f"Analyzing class {class_name} in file {file_path}. \n")
                for i in defname:
                    functions_called = analyze_class_code(class_code, i)
                    ## print(f"functions_called:{functions_called}")
                    if functions_called:
                        check_res = llm_check_NTL(functions_called,i)
                        check_res_2 = extract_call_info(check_res,i,class_name,file_path)

                        results[class_name] = {
                            'file_path': file_path,
                            'functions': functions_called
                        }

    
    return results,check_res_2


def llm_check_NTL_prompt(content, defname):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = f"""After [Code], there is a function’s code, where {defname} is the target function. We want to determine whether there is any lack of fault tolerance when calling the target function. First, carefully read the provided code and identify where {defname} is called. Then, analyze the inputs and outputs of the function when calling {defname}. Note that there may be multiple inputs and outputs. Finally, carefully assess whether any form of fault tolerance exists before and after calling {defname} for both inputs and outputs. Since calling {defname} is prone to errors, we expect all inputs and outputs to have fault tolerance mechanisms. Please return the results in the following JSON format: {{"Method Name": "Please print the name of the method.", "Is there fault tolerance for the input before calling {defname}?": "yes or no","Is there fault tolerance for the output after calling {defname}?": "yes or no", "Detail": "Please provide a detailed analysis"}}"""
    label = """[Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string

 
def llm_check_NTL(content, defname):
    new_content = llm_check_NTL_prompt(content, defname)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results

 
def extract_call_info(str_input, defname,class_name,file_path):
    match = re.search(r'{[\s\S]*}', str_input)  
    with open(result_file, 'a') as f:
        f.write("\n")
        f.write(f"file:{file_path},class:{class_name}.\n")
        if not match:
            f.write("No JSON-like structure found in the input string.\n")
            return None
        json_str = match.group()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            f.write("Failed to parse JSON string.\n")
            return None
        method_name = data.get("Method Name")
        before_def = data.get(f"Is there fault tolerance for the input before calling {defname}?")
        after_def = data.get(f"Is there fault tolerance for the output after calling {defname}?")
        detail_def = data.get("Detail")
        if before_def == "yes" and after_def == "yes":
            f.write(f"No NTL Defects in method {method_name}. \n")
            return True
        else:
            f.write(f"Exist NTL Defects in method {method_name}.Detail: {detail_def} \n\n")
            return False
    


def get_all_class(importProject, projectName):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.typeDecl.name.distinct.l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean



def get_all_method(importProject, projectName):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}");
    cpg.method.name.distinct.l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean



def extract_valid(res1_str):
    res1 = parse_2_string(res1_str)
    invalid_keywords = ["<", ">", "<module>", "<meta>", "<body>", "<metaClassCallHandler>", "<fakeNew>", "<metaClassAdapter>","ANY"]
    valid_class_names = set()
    for item in res1:
        if not any(keyword in item for keyword in invalid_keywords):
            valid_class_names.add(item)
    return list(valid_class_names)



def reorder_class_list(class_names):
    model_related = []
    agent_related = []
    others = []

    for class_name in class_names:
        if any(keyword in class_name for keyword in ["LLM", "llm", "model"]):
            model_related.append(class_name)
        elif any(keyword in class_name for keyword in ["agent", "Agent"]):
            agent_related.append(class_name)
        else:
            others.append(class_name)
    reordered_list = model_related + agent_related + others
    return reordered_list



def get_classes_detail(importProject, projectName, class_names):
    combined_result = ""    
    for class_name in class_names:
        joern_query = f"""
            importCode(inputPath="{importProject}", projectName="{projectName}")
            cpg.typeDecl.name("{class_name}").l
        """
        process = subprocess.Popen(
            ['joern'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(joern_query.encode())
        result = stdout.decode('utf-8')
        result_clean = clean_string(result)
        combined_result += f"\n{result_clean}" 
    ## print(combined_result)
    return combined_result



def extract_functions_from_classes(project_path, class_names):
    class_functions = {class_name: [] for class_name in class_names}
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name in class_names:
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                class_functions[node.name].append(item.name)
    result = {cls: funcs for cls, funcs in class_functions.items() if funcs}
    return result



def filter_and_sort_functions(class_dict, all_functions):

    # Combine all functions from the dictionary into a set for efficient lookup
    excluded_functions = set(func for funcs in class_dict.values() for func in funcs)
    
    # Filter the functions to remove those in the excluded set
    filtered_functions = [func for func in all_functions if func not in excluded_functions]
    
    # Separate functions with 'tool' or 'Tool' and others
    tool_functions = [func for func in filtered_functions if 'tool' in func.lower()]
    other_functions = [func for func in filtered_functions if 'tool' not in func.lower()]
    
    # Concatenate tool functions at the front
    return tool_functions + other_functions


def extract_relevant_json_2(data):
    pattern = re.compile(r'{[^}]+}')
    matches = pattern.findall(data)
    extracted_json_list = []
    for match in matches:
        try:
            json_obj = json.loads(match)         
            if all(key in json_obj for key in ["method Name", "is Tool Implementation Instance", "Tool Name Variable", "Tool Description Variable", "Tool Name Variable Initialization Value", "Tool Description Variable Initialization Value", "Is there an inconsistency between Tool Name, Description, and Implementation", "Detailed Description of the Issue"]):
                extracted_json_list.append(json.dumps(json_obj, indent=2))
        except json.JSONDecodeError:
            continue
    return '\n'.join(extracted_json_list)


def prompt_check_method_tool(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """Following [Code] is the source code of an Agent project along with its folder locations. Typically, an Agent project may include various independent tools. Please carefully review the code and strictly follow the steps below for evaluation:
	Firstly, Analyze each function individually to determine whether it implements a specific independent tool. Strictly evaluate based on the following criteria:
	• The function must implement an independent tool.
	• Exclude functions that, while implementing specific functionalities, do not constitute independent tools.
	• Exclude functions that are related to tools but do not fully implement an independent tool. Secondly,Perform a strict analysis of the code and raise the evaluation standards. Finally, please output the results strictly in the following JSON format, without any additional commentary: {"method Name": "Please print the name of the method", "is Tool Implementation Instance": "yes or no"}\n """
    label = """[Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + analysis_object
    return combined_string


def llm_check_method_tool(content):
    new_content = prompt_check_method_tool(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results



def write_file(content):
    with open(result_file, 'a') as f:
        f.write(f"Method module tool list: {content}.\n")



def get_tool_implementation_methods(input_string):
    json_segments = re.findall(r'\{.*?\}', input_string, re.DOTALL)
    method_names = []
    
    for segment in json_segments:
        try:
            data = json.loads(segment)
            if data.get("is Tool Implementation Instance") == "yes":
                method_names.append(data.get("method Name"))
        except json.JSONDecodeError:
            continue

    return method_names



def main():
    class_info = get_class_info_from_joern(importProject, projectName)
    class_code_location = extract_class_info(class_info)
    class_code_dict,output_data = extract_class_code_1(importProject, class_code_location)
    check_results = process_content("".join(output_data))
    extract_check_results = extract_relevant_json(check_results)
    result_classes = parse_classes(check_results)

    if "Agent Classes" in result_classes:
        agent_class = handle_agent_classes_logic(result_classes, class_code_dict, importProject, "",importProject, projectName)
        agent_class_info = extract_classes_by_name(class_code_dict,agent_class)

    else:
        all_node_str = get_all_class(importProject, projectName)
        all_method_str = get_all_method(importProject, projectName)
        all_node = extract_valid(all_node_str)
        all_method = extract_valid(all_method_str)
        all_class = remove_elements(all_node,all_method)
        all_class_order = reorder_class_list(all_class)
        #all_method_order = reorder_class_list(all_method)
        class_detail = get_classes_detail(importProject, projectName, all_class_order)
        class_code_location_extend = extract_class_info(class_detail)
        class_code_dict_extend,output_data_extend = extract_class_code_1(importProject, class_code_location_extend)
        check_results_extend = process_content("".join(output_data_extend))
        extract_check_results = extract_relevant_json(check_results_extend)
        result_classes_extend = parse_classes(check_results_extend)
        if "Agent Classes" in result_classes_extend:
            agent_class_extend = handle_agent_classes_logic(result_classes_extend, class_code_dict_extend, importProject, "",importProject, projectName)
            agent_class_info = extract_classes_by_name(class_code_dict_extend,agent_class_extend)    
        else:
            dict,agent_method = handle_no_classes_logic(result_classes_extend, importProject, "", class_code_dict_extend,importProject, projectName)
            agent_class_info = extract_classes_by_name(dict,agent_method)

    class_info_1 = get_class_info_from_joern_tool()
    method_info = get_method_info_from_joern()
    new = extract_valid_class_names(class_info_1)
    new1 = extract_valid_class_names(method_info)
    result = remove_elements(new, new1)
    res1 = query_class_inheritance(result)
    inheritance_tree = build_inheritance_tree(res1)
    level_class = level_order_traversal(inheritance_tree)

    res2,res2_code = process_layers(level_class)

    if res2 is None:
        #output_file_1 = f"{importProject}/extracted_functions.txt"
        all_method = extract_user_defined_functions(importProject)
        method_in_class = extract_functions_from_classes(importProject, result)
        #invidial_method = remove_elements(all_method,method_in_class)
        invidial_method = filter_and_sort_functions(method_in_class,all_method)
        extracted_data = extract_specific_functions_from_project(importProject,invidial_method)
        grouped_functions = group_functions_in_batches(extracted_data)
        method_check = ""
        for i in grouped_functions:
            method_check_1 = llm_check_method_tool(i)
            method_check += method_check_1
            method_check += '\n'
        tool_name = extract_relevant_json_3(method_check)  
        tool_list = get_tool_implementation_methods(tool_name)  
        write_file(tool_list)
        method_code,results = extract_function_calls(agent_class_info, tool_list)
    else:
        res3_use_name_llm_check = llm_check_tool_use_class(res2_code)
        res4_use_name = extract_tool_def_info(res3_use_name_llm_check)
        method_code,results = extract_function_calls(agent_class_info, res4_use_name)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_code_new.py <import_project> <project_name>")
        sys.exit(1)

    importProject = sys.argv[1]
    projectName = sys.argv[2]

    print(f"importProject: {importProject} projectName: {projectName}")
    
    main()
