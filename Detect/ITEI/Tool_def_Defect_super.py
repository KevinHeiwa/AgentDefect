import subprocess
import os
import re
import json
import ast
from openai import OpenAI # type: ignore
from collections import deque
import sys


client = OpenAI(
    api_key="",
    base_url=""
)


Model = ""
result_file = ""



def get_class_info_from_joern():

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
    return [item for item in list1 if item not in list2 and ('tool' in item.lower())]


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
        (f"Error: The file '{file_in}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
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
    analysis_message = """After [Class Code], the filename and code for a class is provided. We are looking to find the initial class of the Tool in an Agent project, which other tool instances inherit when implemented. First, carefully read the provided code, then determine whether this code is related to Tool, and further assess whether it is the initial class of Tool. Return the results in the following JSON format: {"Class Name": "Please print the name of the Class.", "Is it related to Tool?": "yes or no?", "Is it the initial class of Tool?": "yes or no?"
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
        print("No JSON-like structure found in the input string.")
        return None
    json_str = match.group()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    is_related_to_tool = data.get("Is it related to Tool?")
    is_initial_class_of_tool = data.get("Is it the initial class of Tool?")
    if is_related_to_tool == "yes" and is_initial_class_of_tool == "yes":
        return data.get("Class Name")
    return None



def process_layers(layers):
    for level in layers:
        for class_name in level:
            details  = query_class_details(class_name)
            
            if details:
                filename, line_number = details[0]                

                class_code = extract_class_code(filename, class_name, line_number)               
                combined_code = f"Filename: {filename}\nCode: {class_code}"
                llm_check_res = llm_check_tool_initialization_class(combined_code)
                find_res = extract_tool_class_info(llm_check_res)
                if find_res is not None:
                    return class_name 

    return None



def get_classextend_info_joern(class_name):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.typeDecl.filter(_.inheritsFromTypeFullName.exists(_.matches(".*{class_name}.*"))).l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    #print(result_clean)
    return result_clean


def extract_type_decl_info(input_str):
    name_pattern = r'\bname\s*=\s*"([^"]+)"'
    filename_pattern = r'\bfilename\s*=\s*"([^"]+)"'
    line_number_pattern = r'\blineNumber\s*=\s*Some\(value\s*=\s*(\d+)\)'
    
    type_decl_blocks = re.split(r'TypeDecl\(', input_str)[1:]  # 去掉第一个空块
    
    result = []
    
    for block in type_decl_blocks:
        name_match = re.search(name_pattern, block)
        filename_match = re.search(filename_pattern, block)
        line_number_match = re.search(line_number_pattern, block)
        
        if name_match and filename_match and line_number_match:
            name = name_match.group(1)
            filename = filename_match.group(1)
            line_number = line_number_match.group(1)
            result.append([name, filename, line_number])
    
    return result


class FunctionCallVisitor(ast.NodeVisitor):

    def __init__(self):
        self.called_functions = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):  
            self.called_functions.add(node.func.id)
        self.generic_visit(node)


def clean_code_block(code_block):

    lines = code_block.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return code_block 

    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    
    cleaned_lines = [line[min_indent:] if len(line.strip()) > 0 else line for line in lines]
    return '\n'.join(cleaned_lines)



def extract_function_code(file_lines, function_name, already_extracted):

    if function_name in already_extracted:
        return ""  

    function_code = []
    inside_function = False
    indent_level = None
    called_functions = set()

    for line in file_lines:
        if line.strip().startswith(f"def {function_name}(") and not inside_function:
            inside_function = True
            indent_level = len(line) - len(line.lstrip())  
            function_code.append(line)
        elif inside_function:
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent > indent_level or line.strip() == '':
                function_code.append(line)
            else:

                break

    function_body = clean_code_block(''.join(function_code))
    try:
        tree = ast.parse(function_body)
    except SyntaxError as e:
        print(f"SyntaxError in function {function_name}: {e}")
        return ""

    visitor = FunctionCallVisitor()
    visitor.visit(tree)
    called_functions.update(visitor.called_functions)
    already_extracted.add(function_name)
    for func in called_functions:
        if func not in already_extracted:
            func_code = extract_function_code(file_lines, func, already_extracted)
            if func_code:
                function_code.append(f"\n# Function definition for {func}\n")
                function_code.append(func_code)

    return ''.join(function_code) if function_code else None



def extract_class_code_extend(filename, class_name, start_line):
    class_code = []
    file_in = os.path.join(importProject, filename)
    try:
        with open(file_in, 'r') as file:
            lines = file.readlines()

        inside_class = False
        indent_level = None
        called_functions = set()  
        already_extracted = set()  
        
        for i, line in enumerate(lines[int(start_line) - 1:], start=int(start_line)):
            if line.strip().startswith(f"class {class_name}") and not inside_class:
                inside_class = True
                indent_level = len(line) - len(line.lstrip())  
                class_code.append(line)
            elif inside_class:
                current_indent = len(line) - len(line.lstrip())

                if current_indent > indent_level or line.strip() == '':
                    class_code.append(line)

                    clean_line = clean_code_block(line)
                    try:
                        tree = ast.parse(clean_line)
                    except SyntaxError as e:
                        print(f"SyntaxError in line {i}: {e}")
                        continue

                    visitor = FunctionCallVisitor()
                    visitor.visit(tree)
                    called_functions.update(visitor.called_functions)
                else:
                    break

        class_methods = ast.parse(''.join(class_code))
        for node in ast.walk(class_methods):
            if isinstance(node, ast.FunctionDef):
                already_extracted.add(node.name)

        for func in called_functions:
            func_code = extract_function_code(lines, func, already_extracted)
            if func_code:
                class_code.append(f"\n# Function definition for {func}\n")
                class_code.append(func_code)

        return ''.join(class_code) if class_code else None
    except FileNotFoundError:
        print(f"Error: The file '{file_in}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def process_class_codes(two_dimensional_array):
    class_codes = []
    
    for class_info in two_dimensional_array:
        class_name = class_info[0]
        filename = class_info[1]
        start_line = int(class_info[2])  
        class_code = extract_class_code_extend(filename, class_name, start_line)
        class_info.append(class_code)
    
    return two_dimensional_array


def parse_llm_check_result(result_str):
    match = re.search(r'{[\s\S]*}', result_str)  
    if not match:
        #print("No JSON-like structure found in the input string.")
        return None
    json_str = match.group()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        #print("Failed to parse JSON string.")
        return None

    return data


def check_tool_inconsistencies(tool_dict, sub_array):
    if tool_dict.get("Are there inconsistencies between the tool's implementation and its name and description?") == "yes":

        inconsistency_flag = tool_dict.get("Are there inconsistencies between the tool's implementation and its name and description?")
        inconsistency_description = tool_dict.get("Please describe the inconsistencies between the tool's implementation and its name and description")

        result = [sub_array[0], sub_array[1], inconsistency_flag, inconsistency_description]
        return result

    if tool_dict.get("Are there variables representing the tool's name and description in the code?") == "no":
        inconsistency_flag = "yes"
        inconsistency_description = "No tool's name and description in code."
        result = [sub_array[0], sub_array[1], inconsistency_flag, inconsistency_description]
        return result
    
    
    if tool_dict.get("Please output the variables representing the tool's description") == "None" or tool_dict.get("Please output the variables representing the tool's description") is None:
        inconsistency_flag = "yes"
        inconsistency_description = "Tool description variable is empty."
        result = [sub_array[0], sub_array[1], inconsistency_flag, inconsistency_description]
        return result

    if tool_dict.get("Please output the initialization values of the variables representing the tool's name") == "None" or tool_dict.get("Please output the initialization values of the variables representing the tool's name") is None:
        inconsistency_flag = "yes"
        inconsistency_description = "Tool name variable initialization is empty."
        result = [sub_array[0], sub_array[1], inconsistency_flag, inconsistency_description]
        return result

    if tool_dict.get("Please output the initialization values of the variables representing the tool's description") == "None" or tool_dict.get("Please output the initialization values of the variables representing the tool's description") is None:
        inconsistency_flag = "yes"
        inconsistency_description = "Tool description variable initialization is empty."
        result = [sub_array[0], sub_array[1], inconsistency_flag, inconsistency_description]
        return result
    else:
        return None
    




def format_tool_info(two_d_array):
    combined_results = []
    
    for sub_array in two_d_array:
        if len(sub_array) >= 2:
            tool_name = sub_array[0]
            tool_code = sub_array[-1]
            result = f"[Tool Name]\n {tool_name}\n[Tool Code]\n{tool_code}\n\n"
            llm_check_result = llm_check_ETE(result)  # Assuming llm_check_ETE returns the JSON-like string

            parsed_result = parse_llm_check_result(llm_check_result)
            #(f"llm check:{parsed_result}",type(parsed_result))

            new_list = check_tool_inconsistencies(parsed_result,sub_array)

            combined_results.append(new_list)
    
    return combined_results



def process_ete_defects(two_dimensional_array):
    with open(result_file, 'a') as f:
        if not two_dimensional_array or all(sub_array is None for sub_array in two_dimensional_array):
            f.write("No ETE defect (Class Module). \n")
            return "No ETE defect. \n"
        
        results = []
        for sub_array in two_dimensional_array:
            if sub_array is not None and len(sub_array) >= 3: 
                class_name = sub_array[0]
                file_name = sub_array[1]
                detail = sub_array[3]
                f.write(f"Exist ETE defect, Class {class_name} in {file_name}, detail:{detail}.\n")
                result = f"Exist ETE defect, Class [{class_name}] in [{file_name}], detail:[{detail}].\n"
                results.append(result)
        
        if not results:  
            f.write("No ETE defect. \n")
            return None
        
    return results

def prompt_ETE(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """Below is the name and source code of some tools in an Agent project. Typically, an Agent project includes external tools to provide additional capabilities. Each tool, in addition to its source code implementation, usually has a name and a description to characterize the tool. You now need to analyze the provided source code to determine if there are inconsistencies between the tool’s implementation, name, and description. Please carefully review the code and strictly follow the criteria below for your analysis:
	1.	Analyze whether the code contains variables representing the tool’s name and description (e.g., name and description).
	2.	If variables representing the tool’s name and description exist, further determine whether these variables are empty.
	3.	Summarize the tool’s implementation code and evaluate whether there are inconsistencies between the tool’s implementation, name, and description. For example:
	•	The tool’s name is “calculator,” but its description states “today’s news.”
	•	Both the name and description indicate the tool is for retrieving weather information, but the implementation is for a search tool. Finally, must return the result only in the following JSON format: {"Tool Name": "Please print the name of the Tool.", "Are there variables representing the tool's name and description in the code?": "yes or no?", "Please output the variables representing the tool's name": "Output variable names (if empty, return None)", "Please output the variables representing the tool's description": "Output variable names (if empty, return None)", "Please output the initialization values of the variables representing the tool's name": "Output the initialization values of the variables (if empty, return None)", "Please output the initialization values of the variables representing the tool's description": "Output the initialization values of the variables (if empty, return None)", "Are there inconsistencies between the tool's implementation and its name and description?": "yes or no?", "Please describe the inconsistencies between the tool's implementation and its name and description": "Describe the inconsistencies (if none, return None)" } \n"""
    analysis_object = content
    combined_string = system_message + analysis_message  + analysis_object
    return combined_string


def llm_check_ETE(content):
    new_content = prompt_ETE(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.completions.create(
        model=,
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results


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

def extract_specific_functions_from_project(project_path, function_names):
    """Extract specified functions from Python files in a project directory and return them in memory."""
    extracted_functions = []  # List to store all extracted function information

    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions = extract_specific_functions_from_file(file_path, function_names)
                if functions:
                    # Add extracted functions to the in-memory list
                    for func in functions:
                        # Combine file path and code for the extracted function
                        combined_code = f"File path: {file_path}\n{func['code']}"
                        extracted_functions.append({
                            "name": func["name"],
                            "code": combined_code
                        })

    return extracted_functions


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
    analysis_message = """Following [Code] is the source code of an Agent project along with its folder locations. Typically, an Agent project may include various independent tools. Please carefully review the code and strictly follow the steps below for evaluation:
	1.	Analyze each function individually to determine whether it implements a specific independent tool. Strictly evaluate based on the following criteria:
	• The function must implement an independent tool.
	• Exclude functions that, while implementing specific functionalities, do not constitute independent tools.
	• Exclude functions that are related to tools but do not fully implement an independent tool.
	2.	For functions identified as implementing independent tools, check whether the code contains variables specifying the tool’s name and a variable describing the tool. If both variables are present, further analyze whether there are inconsistencies between the tool’s name, description, and actual implementation. For example, the tool name might be “calculator,” but the description refers to a real-time weather retrieval tool. Another example is a function’s code that implements a data retrieval tool but has a name like “today’s news.” Note that if the tool’s description is incomplete, such as failing to correctly describe the tool’s functionality, this is also considered an inconsistency between the tool’s name, description, and actual implementation.
	3.	Perform a strict analysis of the code and raise the evaluation standards. Please output only the result in the following JSON format, without additional commentary: {"method Name": "Please print the name of the method.", "is Tool Implementation Instance": "yes or no", "Tool Name Variable": "Please output the variable name (if none, output None)", "Tool Description Variable": "Please output the variable name (if none, output None)", "Tool Name Variable Initialization Value": "Output the initialization value of the tool name variable (if none, output None)", "Tool Description Variable Initialization Value": "Output the initialization value of the tool description variable (if none, output None)", "Is there an inconsistency between Tool Name, Description, and Implementation": "yes or no", "Detailed Description of the Issue": "Specifically describe the identified inconsistency"}\n """
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

def filter_and_sort_functions(class_dict, all_functions):
    """
    Filters out functions from the all_functions list that are present in the class_dict and
    moves functions containing "tool" or "Tool" to the front.

    Args:
        class_dict (dict): A dictionary where keys are class names and values are lists of functions.
        all_functions (list): A list of all functions.

    Returns:
        list: A sorted and filtered list of functions.
    """
    # Combine all functions from the dictionary into a set for efficient lookup
    excluded_functions = set(func for funcs in class_dict.values() for func in funcs)
    
    # Filter the functions to remove those in the excluded set
    filtered_functions = [func for func in all_functions if func not in excluded_functions]
    
    # Separate functions with 'tool' or 'Tool' and others
    tool_functions = [func for func in filtered_functions if 'tool' in func.lower()]
    other_functions = [func for func in filtered_functions if 'tool' not in func.lower()]
    
    # Concatenate tool functions at the front
    return tool_functions + other_functions


def extract_relevant_json_1(data):
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


def methods_ETE(input_string):
    blocks = re.findall(r'\{[^\}]+\}', input_string)
    method_name = re.compile(r'"method Name":\s*"([^"]+)"')
    tool_instance = re.compile(r'"is Tool Implementation Instance":\s*"([^"]+)"')
    tool_name = re.compile(r'"Tool Name Variable":\s*"([^"]+)"')
    tool_description = re.compile(r'"Tool Description Variable":\s*"([^"]+)"')
    tool_name_init = re.compile(r'"Tool Name Variable Initialization Value":\s*"([^"]+)"')
    tool_description_init = re.compile(r'"Tool Description Variable Initialization Value":\s*"([^"]+)"')
    inconsistency = re.compile(r'"Is there an inconsistency between Tool Name, Description, and Implementation":\s*"([^"]+)"')
    detail_description = re.compile(r'"Detailed Description of the Issue":\s*"([^"]+)"')
    with open(result_file, 'a') as f:
        for block in blocks:
            method_names = method_name.findall(block)
            tool_instances = tool_instance.findall(block)
            tool_names = tool_name.findall(block)
            tool_descriptions = tool_description.findall(block)
            tool_name_inits = tool_name_init.findall(block)
            tool_description_inits = tool_description_init.findall(block)
            inconsistencies = inconsistency.findall(block)
            detail_descriptions = detail_description.findall(block)

            method_display = method_names[0] if method_names else 'Unknown method'
            f.write(f"Processing method: {method_display}\n")
            if not tool_instances or 'no' in tool_instances or 'No' in tool_instances or 'NO' in tool_instances:
                f.write(f"Skipping {method_display}: Tool implementation not found.\n")
                continue

            if not tool_names or tool_names[0] == "None":
                if not tool_descriptions or tool_descriptions[0] == "None":
                    f.write(f"Skipping {method_display}: Tool implementation not found.\n\n")
                    continue

            
            if 'yes' in inconsistencies:
                f.write(f"Exist ETE defect: Inconsistency found in {method_display}. ")
                if detail_descriptions:
                    for description in detail_descriptions:
                        if description != "None":
                            f.write(f"Detailed description: {description}.\n\n")
                continue

            f.write(f"No ETE defect found in {method_display}. \n")
            f.write(f"Detailed description: {detail_descriptions}\n\n")


def remove_elements_1(list1, list2):
    return [item for item in list1 if item not in list2]


def main(importProject, projectName):
    class_info = get_class_info_from_joern()
    method_info = get_method_info_from_joern()
    new = extract_valid_class_names(class_info)
    new1 = extract_valid_class_names(method_info)
    result = remove_elements(new, new1)
    result_no_tool_limit = remove_elements_1(new,new1)
    res1 = query_class_inheritance(result)
    inheritance_tree = build_inheritance_tree(res1)
    level_class = level_order_traversal(inheritance_tree)
    res2 = process_layers(level_class)

    if res2 is None:
        all_method = extract_user_defined_functions(importProject)
        method_in_class = extract_functions_from_classes(importProject, result_no_tool_limit)
        invidial_method = filter_and_sort_functions(method_in_class,all_method)
        extracted_data = extract_specific_functions_from_project(importProject,invidial_method)
        grouped_functions = group_functions_in_batches(extracted_data)
        method_check = ""
        for i in grouped_functions:
            method_check_1 = llm_check_method(i)
            method_check += method_check_1
            method_check += '\n'
        No_res = extract_relevant_json_1(method_check)
        methods_ETE(No_res)

    else:
        res3 = get_classextend_info_joern(res2)
        res4 = extract_type_decl_info(res3)
        res5 = process_class_codes(res4)
        res6 = format_tool_info(res5)
        res7 = process_ete_defects(res6)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_code_new.py <import_project> <project_name>")
        sys.exit(1)

    global importProject
    global projectName

    importProject = sys.argv[1]
    projectName = sys.argv[2]
    
    main(importProject, projectName)