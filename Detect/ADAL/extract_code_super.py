import subprocess
import os
import re
import json
import ast
from openai import OpenAI # type: ignore
import sys


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="",
    base_url=""
)

Model_nanme = "gpt-4"
result_file = ""


class ClassAttributeVisitor(ast.NodeVisitor):
    def __init__(self, class_name):
        self.attributes = []
        self.current_class = None
        self.target_class = class_name

    def visit_ClassDef(self, node):
        if node.name == self.target_class:
            self.current_class = node.name
            for class_node in node.body:
                if isinstance(class_node, ast.AnnAssign) or isinstance(class_node, ast.Assign):
                    self.visit(class_node)
            self.current_class = None

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and self.current_class:
            attr_name = node.target.id
            attr_type = ast.unparse(node.annotation)
            attr_value = ast.unparse(node.value) if node.value else 'None'
            self.attributes.append((attr_name, attr_type, attr_value))

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and self.current_class:
            attr_name = node.targets[0].id
            attr_value = ast.unparse(node.value)
            attr_type = 'Any'  # 如果没有显式的类型注解，使用 Any
            self.attributes.append((attr_name, attr_type, attr_value))


def get_all_class(importProject, projectName):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.typeDecl.name.distinct.l
    """
    process = subprocess.Popen([''], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean


def get_all_method(importProject, projectName):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}");
    cpg.method.name.distinct.l
    """
    process = subprocess.Popen([''], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean


def get_class_info_from_joern(importProject, projectName):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.typeDecl.filter(_.inheritsFromTypeFullName.exists(_.matches(".*BaseModel.*"))).l
    """
    process = subprocess.Popen([''], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean


def get_classes_detail(importProject, projectName, class_names):
    combined_result = ""    
    for class_name in class_names:
        joern_query = f"""
            importCode(inputPath="{importProject}", projectName="{projectName}")
            cpg.typeDecl.name("{class_name}").l
        """
        process = subprocess.Popen(
            [''],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(joern_query.encode())
        result = stdout.decode('utf-8')
        result_clean = clean_string(result)
        combined_result += f"\n{result_clean}" 
    return combined_result


def get_method_detail(importProject, projectName, method_names):
    combined_result = ""    
    for method_name in method_names:
        joern_query = f"""
        importCode(inputPath="{importProject}", projectName="{projectName}")
        cpg.method.filter(_.astParentFullName != "<speculatedMethods>").filter(_.filename != "<empty>").filter(_.isExternal == false).filter(_.name == "{method_name}").l
        """
        process = subprocess.Popen(
            [''],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(joern_query.encode())
        result = stdout.decode('utf-8')
        result_clean = clean_string(result)
        combined_result += f"\n{result_clean}" 
    return combined_result


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



def clean_string(input_str):
    sections = re.split(r'joern>', input_str)
    if len(sections) < 3:
        return "Error in extract data from 'joern>'"
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


def extract_code(file_path, start_line, end_line):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return ''.join(lines[start_line - 1:end_line])

def extract_class_code(folder_path, class_info_list):
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


# 判断agent哪些类属性与LLM有关
def prompt_agent_identify(content):
    system_message = """You are a large language model application development engineer, skilled in analyzing elements in the code and the properties of LLMs. \n """
    analysis_message = """Below is the code for an Agent class based on a large model. We aim to find class attributes or variables related to LLM and process them further. Please analyze step by step as follows: First, carefully read the code below and determine whether any variables or class attributes are related to LLM initialization. If none are found, return None. If there are variables or attributes that meet the requirements, further check whether these variables or attributes have non-None and non-empty initialization values. If the variables or attributes have non-None and non-empty initialization values, extract the model name or related information from the initialization values. If the variables or attributes do not have initialization values, or if the initialization values are None or empty, return None. Next, identify whether the variables and attributes are instances of other classes. Finally, return the results strictly in the following JSON format, without any extra characters or content. Note that if there are multiple class attributes or variables that meet the requirements, generate a separate JSON output for each attribute.{ "LLM-related class attributes or variables": "The name of the class attribute or variable that meets the requirements", "LLM-related class attribute or variable initialization value": "The initialization value (if empty or None, return None)", "Is the LLM-related class attribute or variable an instance of another class": "yes or no?", "Model name": "Identify the model's name (if it cannot be identified, return Unknown)", "Which other class instance does the LLM-related class attribute or variable inherit from?": "The name of another class" }\n """
    label = """[Class Agent Code] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string


def llm_check_agent_attr(content):
    new_content = prompt_agent_identify(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=,
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results



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


def prompt_for_llm_chat(content):
    system_message = """You are a developer of large language model applications, capable of accurately identifying elements in the code and clearly distinguishing the attributes of LLMs based on given criteria. \n """
    analysis_message = """Please extract the required attributes or variables based on the given class information. The [Class Information] section contains the class name, its location in the project, and the class source code. First, carefully read the source code of the class to identify which variables or class attributes are related to LLM initialization (e.g., llm, model, etc., where such attributes typically have their initialization values set to the model name or its location). If such variables or attributes exist, extract their initial values, which are the model names. Next, based on existing knowledge and the model name, assess whether the model is a conversational model. For instance, well-known models like the GPT series or the Llama series. If the model cannot be recognized, determine whether its name contains keywords such as “chat.” Furthermore, evaluate whether the model has good generative capabilities, specifically whether it can serve as an Agent’s LLM. Well-known models (e.g., the GPT series) can be assumed to have sufficient generative capabilities. Note that if the model is specialized for a specific task or domain (e.g., code-related models), it should not be considered to have sufficient generative capabilities. For example, CodeLlama is a model specifically designed for code-related tasks, so it is assumed not to have strong generative capabilities. Finally, return the results only in the following JSON format: {"LLM": "The name of the variable or attribute related to LLM initialization (if none, then None)", "LLM Initialization Value": "The initialization value of the LLM variable or attribute (if none, then None)", "LLM Name": "The model's name (if none or cannot be extracted, then None)", "Is the model designed for a specific task/domain?": "yes or no? + Task/domain", "Is the model a Chat model?": "yes or no?", "Does the model have sufficient generative capabilities?": "yes or no?" }\n """
    label = """[Class Information] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string



def llm_check_chat(content):
    #attributes = extract_class_attributes_from_dict(content)
    new_content = prompt_for_llm_chat(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=, 
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results


def prompt_for_agent_llm_chat(content):
    system_message = """You are a developer of large language model applications, capable of accurately identifying elements in the code and clearly distinguishing the attributes of LLMs based on given criteria. \n """
    analysis_message = """Based on the provided LLM information and defined rules, please determine the properties of the LLM. First, extract the name of the LLM from the given information. Then, based on existing knowledge and factors such as the model’s name, assess whether the model is a conversational model. For example, well-known models like the GPT series or Llama series, or models whose names contain keywords like “Chat,” are typically conversational models. Furthermore, based on existing knowledge and the model’s name, determine whether the model has sufficient generative capabilities. Well-known models, such as the GPT series or Llama series, are generally considered to have adequate generative capabilities. It is important to note that if the model is designed for a specific task, such as code-related tasks (often containing the keyword “code” in the name), these models may not be considered to have sufficient generative capabilities. Finally, only return the result must in the following JSON format: { "LLM Name": "The name of the model (if none or cannot be extracted, then None)", "Is the model a Chat model?": "yes or no?", "Does the model have sufficient generative capabilities?": "yes or no?" }\n """
    label = """[LLM information] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string


def agent_llm_check_chat(content):
    new_content = prompt_for_agent_llm_chat(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=, 
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    result = extract_agent_info(results)
    return result


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

    if llm_classes:
        return {"LLM Classes": llm_classes}
    elif agent_classes:
        return {"Agent Classes": agent_classes}
    else:
        return {"No Classes": None}

 
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


    if llm_method:
        return {"LLM Method": llm_method}
    elif agent_method:
        return {"Agent Method": agent_method}
    else:
        return {"No Method": None}
    

def handle_llm_classes(llm_classes, llm_classes_source_dict):
    llm_params = {}
    for file_path, class_code in llm_classes_source_dict.get(llm_classes, {}).items():
        params = extract_class_attributes(llm_classes, class_code)
        llm_params.setdefault(llm_classes, {})[file_path] = params
    return llm_params



def extract_class_attributes(class_names, source_code):
    tree = ast.parse(source_code)
    visitor = ClassAttributeVisitor(class_names)
    visitor.visit(tree)
    #print(f"Extracted attributes for {class_names}: {visitor.attributes}")
    return visitor.attributes
    

def check_amf_defect(data):
    data = str(data)
    if data == '' or not data:
        AMF_results = None
        return AMF_results, None
    
    if isinstance(data, str):
        try:
            clean_data = re.search(r'{[\s\S]*}', data)
            if clean_data is None:
                print("No matching data found.")
                return None, None  
            json_data = clean_data.group()
            data = json.loads(json_data)  
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}") 
            return "Invalid JSON format", None  

    if not isinstance(data, dict):
        return "Input is not a dictionary", None

    if data.get("LLM") is None:
        return None, None

    if data.get("LLM Initialization Value") is None:
        return True, 'LLM Initialization Value is empty.'

    if data.get("Is the model a Chat model?").lower() == "yes" and \
       data.get("Does the model have sufficient generative capabilities?").lower() == "yes":
        return False, None
    else:
        return True, data.get("LLM Name") 



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



def evaluate_agent_llm_data(dict_list):
    if not dict_list or all(d.get("LLM-related class attributes or variables") == 'None' for d in dict_list):
        return "Case 1", None
    for d in dict_list:
        if d.get("Model name") != "Unknown":
            return "Case 2", d.get("Model name")

    inheritance_values = set(
        d.get("Which other class instance does the LLM-related class attribute or variable inherit from?")
        for d in dict_list if d.get("Which other class instance does the LLM-related class attribute or variable inherit from?")
    )
    if inheritance_values:
        return "Case 3", list(inheritance_values)
    return "Case 4", None


def evaluate_llm_info(llm_list):
    if not llm_list or llm_list[0].get('LLM Name') in [None, '']:
        return None, None
    
    llm_info = llm_list[0]
    if llm_info.get('Is the model a Chat model?') == 'yes' and llm_info.get('Does the model have sufficient generative capabilities?') == 'yes':
        return False, llm_info.get('LLM Name')
    else:
        return True, llm_info.get('LLM Name')
    

def extract_class_methods(classes_dict):
    functions = []
    
    for class_name, files in classes_dict.items():
        for file_path, class_code in files.items():
            # Find all function definitions within the class code
            func_defs = re.findall(r'\bdef\s+(\w+)\s*\(', class_code)
            # Append functions along with class name to specify their origin
            functions.extend([f"{func_name}" for func_name in func_defs])
    
    return functions


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
    
    extracted_functions = []  #

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




def handle_case(case, agent_llm_value, class_code_dict, class_name, folder_path, output_file,importProject, projectName):
    """
    """
    with open(result_file, 'a') as f:
        if case == 'Case 1':
            f.write("LLM initialization parameter is None or there is a parsing error")
        elif case == 'Case 4':
            f.write(f"Exist AMF Defect. The type of LLM initialization parameter in {class_name} is None")
        elif case == 'Case 2':
            agent_llm_check_results = agent_llm_check_chat(agent_llm_value)
            agent_llm_check_results_new,LLM_name = evaluate_llm_info(agent_llm_check_results)
            if agent_llm_check_results_new == False:
                f.write("No AMF Defect")
            elif agent_llm_check_results_new == True:
                file_paths = get_file_paths_by_class_name(class_code_dict, class_name)
                f.write(f"Exist AMF Defect. The LLM: {LLM_name} in {class_name} of {file_paths} is not suitable")
            else:
                f.write(f"Unknown AMF result for class {class_name}: {agent_llm_check_results_new}, {agent_llm_check_results}\n")
        elif case == 'Case 3':
            for agent_llm_value_item in agent_llm_value:
                agent_llm_value_location_query = f"""
                importCode(inputPath="{importProject}",projectName="{projectName}")
                cpg.typeDecl.name("{agent_llm_value_item}").l
                """
                process = subprocess.Popen([''], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(agent_llm_value_location_query.encode())                       
                location_results = stdout.decode('utf-8')
                location_cleaned = clean_string(location_results)
                class_info_list = extract_class_info(location_cleaned)
                class_code_dict,log_messages_1 = extract_class_code(folder_path, class_info_list)
                
                file_content = read_file_to_string(log_messages_1)
                check_results = process_content(file_content)
                extract_check_results = extract_relevant_json(check_results)
                # print(f"LLM Check results: {extract_check_results}")
                result_classes = parse_classes(check_results)
                # print(f"Real: {result_classes}")

                if "LLM Classes" in result_classes:
                    handle_llm_classes_logic(result_classes, class_code_dict, folder_path, "",importProject, projectName)
                elif "Agent Classes" in result_classes:
                    handle_agent_classes_logic(result_classes, class_code_dict, folder_path, "",importProject, projectName)
                else:
                    f.write(f"LLM initialization parameter is None or there is a parsing error")
        else:
            f.write(f"Unknown case: {case}")



def handle_llm_classes_logic(result_classes, class_code_dict, folder_path, output_file,importProject, projectName):
    class_names = result_classes["LLM Classes"]
    with open(result_file, 'a') as f:
        if isinstance(class_names, list) and len(class_names) > 0:
            for class_name in class_names:
                try:
                    llm_result = handle_llm_classes(class_name, class_code_dict)
                    check_llm_chat_result = llm_check_chat(llm_result)
                    AMF_results, LLM_name = check_amf_defect(check_llm_chat_result)
                    if AMF_results is False:
                        f.write(f"No AMF Defect for class {class_name}\n")
                    elif AMF_results is True:
                        file_paths = get_file_paths_by_class_name(class_code_dict, class_name)
                        f.write(f"Exist AMF Defect. The LLM: {LLM_name} in {class_name} of {file_paths} is not suitable\n")
                    else:
                        f.write(f"{LLM_name} in {class_name}: {AMF_results}, {check_llm_chat_result}\n")
                except Exception as e:
                    f.write(f"Error processing class {class_name}: {str(e)}\n")
        else:
            f.write(f"No valid LLM Classes found in {result_classes}\n")



def handle_agent_classes_logic(result_classes, class_code_dict, folder_path, output_file,importProject, projectName):
    agent_classes = result_classes["Agent Classes"]
    for class_name in agent_classes:
        if class_name in class_code_dict:
            for file_path, class_code in class_code_dict[class_name].items():
                attributes = extract_class_attributes(class_name, class_code)
                extract_llm_from_gpt = llm_check_agent_attr(class_code)
                extract_llm_info_from_agent_class = extract_agent_info(extract_llm_from_gpt)
                case, agent_llm_value = evaluate_agent_llm_data(extract_llm_info_from_agent_class)
                handle_case(case, agent_llm_value, class_code_dict, class_name, folder_path, "",importProject, projectName)


    

def handle_no_classes_logic(result_classes, folder_path, output_file_1, class_code_dict,importProject, projectName):
    all_method = extract_user_defined_functions(importProject)
    method_in_class = extract_class_methods(class_code_dict)
    invidial_method = remove_elements(all_method,method_in_class)
    order_invidial_method = reorder_class_list(invidial_method)
    extracted_data = extract_specific_functions_from_project(importProject,order_invidial_method)
    grouped_functions = group_functions_in_batches(extracted_data)
    #print(grouped_functions)
    method_check = ""
    for i in grouped_functions:
        method_check_1 = llm_check_method(i)
        method_check += method_check_1
        method_check += '\n'
    No_res = extract_relevant_json_1(method_check)
    res_method = parse_methods(No_res)
    # res_method = {'LLM Method': ['_get_default_python_repl']}
    if "LLM Method" in res_method:
        with open(result_file, 'a') as f:
            function_names = res_method.get("LLM Method", [])
            for func_name in function_names:
                for func_data in extracted_data:
                    if func_data["name"] == func_name:
                        path = func_data["path"]
                        code = func_data["code"]
                        #print(f"Found function '{func_name}' in file: {path}")
                        # Call the llm_check_attri function with the extracted code
                        res_1 = llm_check_method_attri(code)
                        AMF_results, LLM_name = check_amf_defect(res_1)
                        if AMF_results is False:
                            f.write(f"No AMF Defect for method {func_name}\n")
                        elif AMF_results is True:
                            f.write(f"Exist AMF Defect. The LLM: {LLM_name} in {func_name} of {path} is not suitable\n")
                        else:
                            f.write(f"Unknow AMF result for method {func_name}: {AMF_results}, {res_1}\n")
                    else:
                        f.write(f"Unknow AMF result, No LLM attributes in extracted_data")
        #handle_llm_classes_logic(res_method, class_code_dict, importProject, output_file,importProject, projectName)
    elif "Agent Method" in res_method:
        with open(result_file, 'a') as f:
            function_names = res_method.get("Agent Method", [])
            for func_name in function_names:
                for func_data in extracted_data:
                    if func_data["name"] == func_name:
                        path = func_data["path"]
                        code = func_data["code"]
                        f.write(f"Found function '{func_name}' in file: {path}")
                        # Call the llm_check_attri function with the extracted code
                        res_1 = llm_check_method_attri(code)
                        AMF_results, LLM_name = check_amf_defect(res_1)
                        if AMF_results is False:
                            f.write(f"No AMF Defect for method {func_name}\n")
                        elif AMF_results is True:
                            f.write(f"Exist AMF Defect. The LLM: {LLM_name} in {func_name} of {path} is not suitable\n")
                        else:
                            f.write(f"Unknow AMF result for method {func_name}: {AMF_results}, {res_1}\n")
                    else:
                        f.write(f"Unknow AMF result, No LLM attributes in extracted_data")
    else:
        with open(result_file, 'a') as f:
            f.write(f"Unknow AMF result, No LLM attributes in extracted_data")

        #handle_no_classes_logic(result_classes, importProject, output_file_1, class_code_dict,importProject, projectName)

   
def main():
    class_info = get_class_info_from_joern(importProject, projectName)
    class_code_location = extract_class_info(class_info)
    class_code_dict,output_data = extract_class_code(importProject, class_code_location)
    check_results = process_content("".join(output_data))
    extract_check_results = extract_relevant_json(check_results)
    result_classes = parse_classes(check_results)

    if "LLM Classes" in result_classes:
        handle_llm_classes_logic(result_classes, class_code_dict, importProject, "", importProject, projectName)
    elif "Agent Classes" in result_classes:
        handle_agent_classes_logic(result_classes, class_code_dict, importProject, "", importProject, projectName)
    else:
        all_node_str = get_all_class(importProject, projectName)
        all_method_str = get_all_method(importProject, projectName)
        all_node = extract_valid(all_node_str)
        all_method = extract_valid(all_method_str)
        all_class = remove_elements(all_node, all_method)
        all_class_order = reorder_class_list(all_class)
        class_detail = get_classes_detail(importProject, projectName, all_class_order)
        class_code_location_extend = extract_class_info(class_detail)
        class_code_dict_extend,output_data_extend = extract_class_code(importProject, class_code_location_extend)
        check_results_extend = process_content("".join(output_data_extend))
        extract_check_results = extract_relevant_json(check_results_extend)
        result_classes_extend = parse_classes(check_results_extend)
        if "LLM Classes" in result_classes_extend:
            handle_llm_classes_logic(result_classes_extend, class_code_dict_extend, importProject, "", importProject, projectName)
        elif "Agent Classes" in result_classes_extend:
            handle_agent_classes_logic(result_classes_extend, class_code_dict_extend, importProject, "", importProject, projectName)
        else:
            handle_no_classes_logic(result_classes_extend, importProject, "", class_code_dict_extend, importProject, projectName)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_code_new.py <import_project> <project_name>")
        sys.exit(1)

    global importProject
    global projectName
    importProject = sys.argv[1]
    projectName = sys.argv[2]
    
    main()
