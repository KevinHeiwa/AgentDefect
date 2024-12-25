import subprocess
import os
import re
import json
import ast
from openai import OpenAI # type: ignore
import sys

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


client = OpenAI(
    api_key=,
    base_url=""
)


Model = ""
result_file = ""


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
    # print(result_clean)
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


def extract_class_code(folder_path, class_info_list):
    class_code_dict = {}  
    log_messages = []  

    for class_info in class_info_list:
        file_path = os.path.join(folder_path, class_info['filename'])          
        if not os.path.exists(file_path):
            log_messages.append(f"File {file_path} not found.")
            continue
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()

        class_name = class_info['classname']
        class_start = re.search(rf'class {class_name}\b', source_code)
        if not class_start:
            log_messages.append(f"Class {class_name} not found in {file_path}.")
            continue

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            log_messages.append(f"Error parsing {file_path}: {e}")
            continue

        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break
        if not class_node:
            log_messages.append(f"Class {class_name} not found in {file_path}.")
            continue

        class_code_lines = source_code.splitlines()[class_node.lineno - 1:class_node.end_lineno]
        class_code = "\n".join(class_code_lines)

        if class_name not in class_code_dict:
            class_code_dict[class_name] = {}
        class_code_dict[class_name][file_path] = class_code

        combined_string = f"Class {class_name} from {file_path}:\n{class_code}\n\classend\n"
        log_messages.append(combined_string)

    return class_code_dict, log_messages



def prompt(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """Please analyze the following code to determine whether it meets the requirements. The code includes several classes along with their respective locations. Analyze whether each of the following classes is an Agent initialization class or an LLM initialization class, and only output the result must in the following JSON format: { "Class Name": "Please print the name of the Class.", "Initialization class of Agent": "yes or no?", "Initialization class of LLM": "yes or no?" } \n """
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


def parse_classes_for_llm(input_string):

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

    if llm_classes:
        return {"LLM Classes": llm_classes}
    else:
        return None
    


def get_file_paths_by_class_name(class_code_dict, class_name):
    if class_name in class_code_dict:
        # 获取所有file_path
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
    print(f"Agent class is {result_classes}")
    # print(f"{class_code_dict}")
    agent_classes = extract_agent_classes(result_classes)
    ## print(f"{agent_classes}")
    return agent_classes




def extract_llm_classes(result_classes):
    if 'LLM Classes' in result_classes:
        agent_classes = result_classes['LLM Classes']
        return agent_classes
    else:
        return []



def handle_llm_classes_logic(result_classes, class_code_dict, folder_path, output_file,importProject, projectName):
    # print(f"LLM class is {result_classes}")
    ## print(f"{class_code_dict}")
    agent_classes = extract_llm_classes(result_classes)
    ## print(f"{agent_classes}")
    return agent_classes



def extract_classes_by_name(class_dict, agent_class):

    
    if class_dict is None or agent_class is None:
        # print("class_dict is None")
        return {}

    extracted_classes = {}

    for class_name in agent_class:
        if class_name in class_dict:
            extracted_classes[class_name] = class_dict[class_name]
        else:
            print(f"Class {class_name} not found.")
    
    return extracted_classes


def llm_check_llm_use_prompt(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """After [Class Code], there is code for an LLM class, and this class is called every time the LLM is invoked. This class contains many functions, and we want to find the function that is used to invoke the LLM externally. In other words, when the LLM is called, this function in the class is directly invoked. First, carefully read the class code and the functions it contains. Next, identify the function that meets this criterion. Finally, return the results in the following JSON format: {"Class Name": "Please print the name of the Class.", "Name of the function that meets the criteria": "Please output the function name (if none, return None)"}"""
    label = """[Class Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string


# 使用LLM check哪些函数或类与Agent初始化或LLM初始化有关

def llm_check_llm_use_class(content):
    new_content = llm_check_llm_use_prompt(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results


def write_file(file_path,class_name):
    with open(result_file, 'a') as f:
        f.write(f"File path: {file_path}")
        f.write(f"Processing class: {class_name}")



def extract_class_code_1(class_dict):
    class_code_results = {}
    res2 = [] 
    for class_name, class_info in class_dict.items():
        for file_path, class_code in class_info.items():
            write_file(file_path,class_name)
            res = llm_check_llm_use_class(class_code)
            extracted_info = extract_llm_def_info(res)
            if extracted_info is not None:
                res2.append(extracted_info)
            class_code_results[class_name] = {
                'file_path': file_path,
                'code': class_code
            }
            
    return res2 #,class_code_results


def extract_llm_def_info(str_input):
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
    class_name = data.get("Class Name")
    use_def = data.get("Name of the function that meets the criteria")
    if use_def is not None or use_def != "None":
        return use_def

    return None


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



def llm_check_PE_prompt(content, defname):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = f"""After [Code], there is a function’s code, where {defname} is the target function. We want to determine whether there is any lack of fault tolerance when calling the target function. First, carefully read the provided code and identify where {defname} is called. Then, analyze the inputs and outputs of the function when calling {defname}. Note that there may be multiple inputs and outputs. Finally, carefully assess whether any form of fault tolerance exists before and after calling {defname} for both inputs and outputs. Since calling {defname} is prone to errors, we expect all inputs and outputs to have fault tolerance mechanisms. Please return the results in the following JSON format: {{"Method Name": "Please print the name of the method.", "Is there fault tolerance for the input before calling {defname}?": "yes or no","Is there fault tolerance for the output after calling {defname}?": "yes or no", "Detail": "Please provide a detailed analysis"}}"""
    label = """[Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string


def llm_check_PE(content, defname):
    new_content = llm_check_PE_prompt(content, defname)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=)
    results = response.choices[0].message.content
    return results


def write_file_1(file_path,class_name):
    with open(result_file, 'a') as f:
        f.write(f"Analyzing class {class_name} in file {file_path}")


def extract_function_calls(class_dict, defname):
    results = {}
    check_res_2 = []
    unique_defnames = list(set(defname))

    for class_name, file_info in class_dict.items():
        for file_path, class_code in file_info.items():
            for func_name in unique_defnames:
                write_file_1(file_path, class_name)
                functions_called = analyze_class_code(class_code, func_name)

                if functions_called:
                    check_res = llm_check_PE(functions_called, func_name)
                    check_res_1 = extract_call_info(check_res, func_name, class_name, file_path)
                    if check_res_1:
                        check_res_2.append(check_res_1)
                    results[class_name] = {
                        'file_path': file_path,
                        'functions': functions_called
                    }

    return results, check_res_2


def extract_call_info(str_input, defname,class_name,file_path):
    match = re.search(r'{[\s\S]*}', str_input)  
    with open(result_file, 'a') as f:
        f.write("\n")
        f.write(f"file:{file_path}, class:{class_name}. \n")
        if not match:
            f.write("No JSON-like structure found in the input string. \n")
            return None
        json_str = match.group()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            f.write("Failed to parse JSON string. \n")
            return None
        method_name = data.get("Method Name")
        before_def = data.get(f"Is there fault tolerance for the input before calling {defname}?")
        after_def = data.get(f"Is there fault tolerance for the output after calling {defname}?")
        detail_def = data.get("Detail")
        if before_def == "yes" and after_def == "yes":
            f.write(f"No PE Defects in method{method_name}. \n")
            return True
        else:
            f.write(f"Exist PE Defects in method {method_name}. Detail: {detail_def}. \n\n")
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


def parse_2_string(res1_str):
    res1_list = re.findall(r'"(.*?)"|\'(.*?)\'', res1_str)
    res1 = [item for sublist in res1_list for item in sublist if item]
    return res1


def extract_valid(res1_str):
    res1 = parse_2_string(res1_str)
    invalid_keywords = ["<", ">", "<module>", "<meta>", "<body>", "<metaClassCallHandler>", "<fakeNew>", "<metaClassAdapter>","ANY"]
    valid_class_names = set()
    for item in res1:
        if not any(keyword in item for keyword in invalid_keywords):
            valid_class_names.add(item)
    return list(valid_class_names)


def remove_elements(list1, list2):
    return [item for item in list1 if item not in list2]



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
    return combined_result



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

    #with open(output_file, "w", encoding="utf-8") as output:
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions = extract_specific_functions_from_file(file_path, function_names)
                if functions:
                    for func in functions:
                        # Combine file path and code for the extracted function
                        combined_code = f"File path: {file_path}\n{func['code']}"
                        extracted_functions.append({
                            "name": func["name"],
                            "code": combined_code
                        })
        # print(f"Specified functions have been saved to {output_file}")

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


# 使用LLM check哪些函数或类与Agent初始化或LLM初始化有关
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

def parse_methods_llm(input_string):
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

    if llm_method:
        return {"LLM Method": llm_method}
    # elif agent_method:
    #     return {"Agent Method": agent_method}
    else:
        return {"No Method": None}
    
def parse_methods_agent(input_string):
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

def handle_no_classes_logic(tag,result_classes, folder_path,output_file_1,  class_code_dict,importProject, projectName):
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

    if tag == True or tag is True:
        res_method = parse_methods_agent(No_res)

        if "Agent Method" in res_method:
            #with open(result_file, 'a') as f:
            function_names = res_method.get("Agent Method", [])

            return function_names
        else:
            return None
    else:
        res_method = parse_methods_llm(No_res)

        if "LLM Method" in res_method:
            #with open(result_file, 'a') as f:
            function_names = res_method.get("LLM Method", [])

            return function_names
        else:
            return None







def main():
    class_info = get_class_info_from_joern(importProject, projectName)
    class_code_location = extract_class_info(class_info)
    class_code_dict,output_data = extract_class_code(importProject, class_code_location)
    check_results = process_content("".join(output_data))
    extract_check_results = extract_relevant_json(check_results)
    result_classes = parse_classes(check_results)
    result_classes_llm = parse_classes_for_llm(check_results)



    if result_classes_llm and "LLM Classes" in result_classes_llm:
        llm_class = handle_llm_classes_logic(result_classes_llm, class_code_dict, importProject, "" ,importProject, projectName)
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
        result_classes_llm_extend = parse_classes_for_llm(check_results_extend)
        if result_classes_llm_extend and "LLM Classes" in result_classes_llm_extend:
            llm_class = handle_llm_classes_logic(result_classes_llm_extend, class_code_dict_extend, importProject, "" ,importProject, projectName)
        else:
            llm_class = handle_no_classes_logic(False,result_classes_extend, importProject, "", class_code_dict_extend,importProject, projectName)


    if "Agent Classes" in result_classes:
        agent_class = handle_agent_classes_logic(result_classes, class_code_dict, importProject, "",importProject, projectName)
    else:
        all_node_str_2 = get_all_class(importProject, projectName)
        all_method_str_2 = get_all_method(importProject, projectName)
        all_node_2 = extract_valid(all_node_str_2)
        all_method_2 = extract_valid(all_method_str_2)
        all_class_2 = remove_elements(all_node_2,all_method_2)
        all_class_order_2 = reorder_class_list(all_class_2)
        #all_method_order = reorder_class_list(all_method)
        class_detail_2 = get_classes_detail(importProject, projectName, all_class_order_2)
        class_code_location_extend_2 = extract_class_info(class_detail_2)
        class_code_dict_extend_2,output_data_extend_2 = extract_class_code_1(importProject, class_code_location_extend_2)
        check_results_extend_2 = process_content("".join(output_data_extend_2))
        extract_check_results_2 = extract_relevant_json(check_results_extend_2)
        result_classes_extend_2 = parse_classes(check_results_extend_2)
        #result_classes_llm_extend_2 = parse_classes_for_llm(check_results_extend_2)
        if "Agent Classes" in result_classes_extend_2:
            agent_class = handle_agent_classes_logic(result_classes_extend_2, class_code_dict_extend_2, importProject, "",importProject, projectName)
        else:
            agent_class = handle_no_classes_logic(True, result_classes_extend_2, importProject, "", class_code_dict_extend_2,importProject, projectName)
    
    
    agent_class_info = extract_classes_by_name(class_code_dict,agent_class)
    llm_class_info = extract_classes_by_name(class_code_dict,llm_class)
    res2 = extract_class_code_1(llm_class_info)
    method_code,results = extract_function_calls(agent_class_info, res2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_code_new.py <import_project> <project_name>")
        sys.exit(1)

    importProject = sys.argv[1]
    projectName = sys.argv[2]
    
    main()
