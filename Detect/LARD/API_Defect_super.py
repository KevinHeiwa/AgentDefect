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


Model = ""
result_file = ""



def get_class_info_from_joern(importProject, projectName):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}")
    cpg.typeDecl.filter(_.inheritsFromTypeFullName.exists(_.matches(".*BaseModel.*"))).l
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


def read_file_to_string(log_messages):
    return "\n".join(log_messages)
    


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
    response = client.chat.completions.create(
        model=ChatAnyWhere_Model,
        messages=messages,
        temperature=0.6,)
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
        return {"All Classes": all_classes}
    


def handle_llm_classes(llm_classes, llm_classes_source_dict):
    llm_functions = {}
    for file_path, class_code in llm_classes_source_dict.get(llm_classes, {}).items():
        functions = extract_class_functions(llm_classes, class_code)
        print(f"Class name '{llm_classes}' in file '{file_path}' has functions: {functions}")
        llm_functions.setdefault(llm_classes, {})[file_path] = functions
    return llm_functions


def extract_class_functions(class_names, source_code):
    tree = ast.parse(source_code)
    visitor = ClassFunctionVisitor(class_names, source_code)
    visitor.visit(tree)
    return visitor.functions


class ClassFunctionVisitor(ast.NodeVisitor):
    def __init__(self, class_name, source_code):
        self.functions = []
        self.current_class = None
        self.target_class = class_name
        self.source_code = source_code.splitlines()


    def visit_ClassDef(self, node):    
        if node.name == self.target_class:
            self.current_class = node.name
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    self.visit(class_node)
            self.current_class = None

    def visit_FunctionDef(self, node):
        if self.current_class:
            func_name = node.name
            start_line = node.lineno - 1
            end_line = node.end_lineno
            func_code = "\n".join(self.source_code[start_line:end_line])
            self.functions.append((func_name, func_code))


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
            attr_type = 'Any' 
            self.attributes.append((attr_name, attr_type, attr_value))


# 判断哪些类与LLM的初始化相关
def prompt_for_llm_chat(content):
    system_message = """You are a developer of large language model applications, capable of accurately identifying elements in the code and clearly distinguishing the attributes of LLMs based on given criteria. \n """
    analysis_message = """Based on the provided class and function information, please perform the relevant function analysis. The information includes the class, its file location, and all functions within the class along with their source code. The analysis requires the following judgments: 1. Carefully read and inspect the class code and its functions to determine whether the class contains operations that involve calling an LLM. 2. If there are operations involving LLM calls, further determine whether the calls are made to a local model or via an API. Common API-based models include OpenAI models and Claude models. 3. Finally, check whether the code calling the model contains any errors, such as missing model names or critical elements (e.g., prompts). Please note that it is essential to carefully verify whether the code calling the LLM API is correct. If parameters such as api_key, model name, or stop tokens are missing, or if these parameters exist but have empty or None values, they should be considered incorrect. Additionally, please note to ignore errors where a variable is not defined within the provided code snippet. Below are some common standard calling patterns to pay special attention to: OpenAI API Call: from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "write a haiku about ai"}
    ]
)
Claude API Call:
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="sk-ant-api03-mrmRzF-1x397AtlRNiur-PEQ7TlpAf3HxY4YQXeyvfB1p642QIAiMYN4Qwv98mKPCJOhA-ZvoYKgAA",
)
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
Gemini API Call:
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
Carefully identify whether the LLM API calls in the code are correct and check for missing required parameters or incorrect initialization values for these parameters. Please return the result in the following JSON format:
{
  "Does the code contain functionality that calls an LLM?": "yes or no?",
  "Method of calling the model": "API or Local? (if none, return None)",
  "Does the LLM call contain any defects?": "yes or no",
  "Is there a lack of initialization for necessary elements during the call, or are the necessary elements initialized to empty or null values?": "yes or no",
  "Defect information": "Output the defect information when calling the model (if none, return None)"
}\n """
    label = """[Class and function Information] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string


def llm_check_chat(content):
    new_content = prompt_for_llm_chat(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=, 
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results


def check_LARL_defect(data):
    clean_data = re.search(r'{[\s\S]*}', data)
    if not clean_data:
        return None, None,None
    json_data = clean_data.group()
    try:
        parsed_data = json.loads(json_data)
    except json.JSONDecodeError:
        return None, None,None
    if parsed_data.get("Method of calling the model") == "API":
        call_way = "API"
    elif parsed_data.get("Method of calling the model") == "Local":
        call_way = "Local"
    else:
        call_way = "None"
    defect_description = parsed_data.get("Defect information")
    if parsed_data.get("Does the code contain functionality that calls an LLM?") == "no":
        return None, None,False, defect_description
    elif parsed_data.get("Does the code contain functionality that calls an LLM?") == "yes":
        if parsed_data.get("Does the LLM call contain any defects?") == "no":
            if parsed_data.get("Is there a lack of initialization for necessary elements during the call, or are the necessary elements initialized to empty or null values?") == "yes":
                return False, call_way, True, defect_description
            else:
                return False, call_way, False, defect_description
        elif parsed_data.get("Does the LLM call contain any defects?") == "yes":
            if parsed_data.get("Is there a lack of initialization for necessary elements during the call, or are the necessary elements initialized to empty or null values?") == "yes":
                return True, call_way, True, defect_description
            else:
                return True, call_way, False, defect_description

    return None,None,None,defect_description  


def get_file_paths_by_class_name(class_code_dict, class_name):
    if class_name in class_code_dict:
        file_paths = class_code_dict[class_name].keys()
        return list(file_paths)
    else:
        return f"Class '{class_name}' not exit"


def prompt_for_llm_initialization(content):
    system_message = """You are a developer of large language model applications. You can accurately identify whether there is an LLM initialization function within a class and extract relevant properties based on given criteria. \n """
    analysis_message = """Based on the provided information under [Class and function Information], please perform the relevant analysis. The provided information includes the class, its file location, as well as all functions within the class and their source code. You need to make the following judgments: 1. Carefully read and inspect the class code and its functions to determine whether the class contains operations that involve calling an LLM. 2. If there are operations involving LLM calls, further determine whether the calls are made to a local model or via an API. Common API-based models include OpenAI models and Claude models. 3. Finally, check whether the code calling the model contains any errors, such as missing model names or critical elements (e.g., prompts). Please note that it is essential to carefully verify whether the code calling the LLM API is correct. If parameters such as api_key, model name, or stop tokens are missing, or if these parameters exist but have empty or None values, they should be considered incorrect. Additionally, please note to ignore errors where a variable is not defined within the provided code snippet. Below are some common standard calling patterns to pay special attention to:
OpenAI API Call Example: 
[Example]
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "write a haiku about ai"}
    ]
)
[Example End]
Claude API Call Example:
[Example]
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="sk-ant-api03-mrmRzF-1x397AtlRNiur-PEQ7TlpAf3HxY4YQXeyvfB1p642QIAiMYN4Qwv98mKPCJOhA-ZvoYKgAA",
)
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
[Example End]
Gemini API Call Example:
[Example]
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
[Example End]
Huggingface Call Example:
[Example]
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
)
[Example End]
Local LLM Call Example:
[Example]
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,  device_map='auto', mirror='tuna').to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
[Example End]
Carefully identify whether the LLM API calls in the code are correct and check for missing required parameters or incorrect initialization values for these parameters. Please return the result in the following JSON format:
{
  "Does the code contain functionality that calls an LLM?": "yes or no?",
  "Method of calling the model": "API or Local? (if none, return None)",
  "Does the LLM call contain any defects?": "yes or no",
  "Is there a lack of initialization for necessary elements during the call, or are the necessary elements initialized to empty or null values?": "yes or no",
  "Defect information": "Output the defect information when calling the model (if none, return None)"
}\n  """
    
    label = """[Class and function Information] \n"""
    analysis_object = content  
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string


def llm_check_initialization(content):
    new_content = prompt_for_llm_initialization(content) 
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
        model=, 
        messages=messages,
        temperature=,)
    results = response.choices[0].message.content
    return results



def extract_class_attributes(class_names, source_code):
    tree = ast.parse(source_code)
    visitor = ClassAttributeVisitor(class_names)
    visitor.visit(tree)
    #print(f"Extracted attributes for {class_names}: {visitor.attributes}")
    return visitor.attributes



def prompt_agent_identify(content):
    system_message = """You are a large language model application development engineer, skilled in analyzing elements in the code and the properties of LLMs. \n """
    analysis_message = """Below are all the class attributes in the Agent class. Please analyze them step by step as follows: First, analyze whether any of these attributes are related to LLM initialization. If none are related, return None. Next, identify the types of these attributes, and recognize which attributes are instances of other classes. Finally, return only the following results, strictly in the JSON format below. Do not return any extra characters or content. Note, if there are multiple class attributes that meet the requirements, the JSON output will be generated for each attribute one by one.{
  "LLM-related class attributes": "The name of the class attribute that meets the requirements",
  "LLM-related class attribute an instance of another class": "yes or no?",
  "Which other class instance does the LLM-related class attribute inherit from?": "The name of another class"
}\n """
    label = """[Attribute in Class Agent] \n"""
    analysis_object = content
    combined_string = system_message + analysis_message + label + str(analysis_object)
    return combined_string



def llm_check_agent_attr(content):
    new_content = prompt_agent_identify(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.completions.create(
        model=ChatAnyWhere_Model,
        messages=messages,
        temperature=0.7,)
    results = response.choices[0].message.content
    return results



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
    if not dict_list or all(d.get("LLM-related class attribute") == 'None' for d in dict_list):
        return "Case 1", None
    inheritance_values = set(
        d.get("Which other class instance does the LLM-related class attribute inherit from?")
        for d in dict_list if d.get("Which other class instance does the LLM-related class attribute inherit from?")
    )
    if inheritance_values:
        return "Case 2", list(inheritance_values)
    return "Case 3", None


def extract_variables_from_result(joern_result):
    pattern = r'argumentName = Some\(value = "(\w+)"\),\s+code = """(.*?)"""|argumentName = Some\(value = "(\w+)"\),\s+code = "([^"]+)"'

    variables = []
    argument = []
    for match in re.finditer(pattern, joern_result, re.DOTALL):
        if match.group(1) and match.group(2):  
            argument_name = match.group(1)
            code = match.group(2).strip()
        elif match.group(3) and match.group(4):  
            argument_name = match.group(3)
            code = match.group(4).strip()
        else:
            continue

        variables.append((argument_name, code))
        argument.append(argument_name)
    
    return variables,argument
    


def query_variables_in_function(class_name, function_name):

    
    joern_query = f"""
    importCode(inputPath="{importProject}", projectName="{projectName}")
    cpg.method.name("{function_name}").where(_.typeDecl.name("{class_name}")).ast.isCall.argument.l
    """
    
    try:
        process = subprocess.Popen(['joern'], 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(joern_query.encode())
        
        
        # print(f"Error: {stderr.decode('utf-8')}")
        result = stdout.decode('utf-8')
        # print("Query Result:\n", result)
        result_clean = clean_string(result)
        # print(result_clean)
        return result_clean

    except Exception as e:
        print(f"Exception occurred: {str(e)}")





def filter_identifiers(input_str):

    filtered_codes = []
    lines = input_str.split('\n')
    current_node = {}
    is_identifier = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('Identifier('):
            current_node = {}
            is_identifier = True
            continue

        if not is_identifier:
            continue

        if line == '),':
            if (current_node.get('argumentName') == 'None' and 
                current_node.get('typeFullName', '').startswith('__builtin')):
                if current_node.get('code'):
                    code = current_node['code'].strip('"')
                    filtered_codes.append(code)
            is_identifier = False
            continue
            
        if '=' in line:
            key, value = [x.strip() for x in line.split('=', 1)]
            value = value.strip(',')
            if key == 'argumentName':
                current_node['argumentName'] = value
            elif key == 'code':
                current_node['code'] = value.strip('"')
            elif key == 'typeFullName':
                current_node['typeFullName'] = value.strip('"')
    
    return filtered_codes



class FunctionParser(ast.NodeVisitor):
    def __init__(self):
        self.variables_and_arguments = []

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            self.variables_and_arguments.append(arg.arg)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            self.variables_and_arguments.append(var_name)
    
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                self.variables_and_arguments.append(var_name)

def parse_function(code):
    tree = ast.parse(code)
    parser = FunctionParser()
    parser.visit(tree)

    return parser.variables_and_arguments
    


def clean_list(list1, list2):
    meaningless_prefixes = ['tmp']
    cleaned_list = [
        item for item in list1
        if item not in list2 and not any(item.startswith(prefix) for prefix in meaningless_prefixes)
    ]
    
    return cleaned_list

def query_joern(class_name, variable_name):
    
    joern_query = f"""
    importCode(inputPath="{importProject}", projectName="{projectName}")
    cpg.typeDecl.name("{class_name}").assignment.target.code("{variable_name}").l
    """
    
    try:
        # 启动 Joern 进程并执行查询
        process = subprocess.Popen(['joern'], 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(joern_query.encode())
        
        result = stdout.decode('utf-8')
        result_clean = clean_string(result)
        return result_clean

    except Exception as e:
        # print(f"Exception occurred: {str(e)}")
        return ""



def process_variable_pairs(pairs, class_name):
    output_list = []
    
    for pair in pairs:
        variable_name, variable_value = pair
        if variable_value.startswith("self."):
            extracted_value = variable_value.split("self.")[1]
            joern_result = query_joern(class_name, extracted_value)
            if not joern_result:
                output_list.append(variable_name)
    
    return output_list




def attr_flow_in(class_name,function_name,code):


    res = query_variables_in_function(class_name, function_name)
    #print(type(res))
    res1, res2 = extract_variables_from_result(res)
    res3 = filter_identifiers(res)
    res4 = parse_function(code.strip())
    res5 = clean_list(res3,res4)
    res6 = process_variable_pairs(res1, class_name)
    combined_list = res5 + res6
    if len(combined_list) == 0:
        return False, None
    else:
        return True, combined_list


def extract_function_code(class_dict, target_class_name, target_function_name):
    if target_class_name in class_dict:
        file_path, class_code = list(class_dict[target_class_name].items())[0]
    else:
        raise ValueError(f"Class {target_class_name} not found in the provided dictionary.")
    lines = class_code.split('\n')
    function_code = []
    inside_function = False
    indent_level = None
    for line in lines:
        if line.strip().startswith(f"def {target_function_name}("):
            inside_function = True
            indent_level = len(line) - len(line.lstrip())
            function_code.append(line)
        elif inside_function:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip():
                break
            function_code.append(line)

    if not function_code:
        return None
    return "\n".join(function_code)



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


def parse_2_string(res1_str):
    res1_list = re.findall(r'"(.*?)"|\'(.*?)\'', res1_str)
    res1 = [item for sublist in res1_list for item in sublist if item]
    return res1


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
    """Extract specified functions from Python files in a project directory and save them to file and a dictionary."""
    extracted_functions = []  # List to store all extracted function information

    #with open(output_file, "w", encoding="utf-8") as output:
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions = extract_specific_functions_from_file(file_path, function_names)
                if functions:

                    # Add to the extracted functions list
                    for func in functions:
                        # Combine file path and code for the extracted function
                        combined_code = f"File path: {file_path}\n{func['code']}"
                        extracted_functions.append({
                            "name": func["name"],
                            "code": combined_code
                        })
        # print(f"Specified functions have been saved to {output_file}")

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



def prompt_for_check_method_attri(content):
    system_message = """You are a developer of large language model applications, capable of accurately identifying elements in the code and clearly distinguishing the attributes of LLMs based on given criteria. \n """
    analysis_message = """Please analyze the following functions’ code based on the instructions below. You need to make the following judgments: 1. Carefully read and functions to determine whether the contains operations that involve calling an LLM. 2. If there are operations involving LLM calls, further determine whether the calls are made to a local model or via an API. Common API-based models include OpenAI models and Claude models. 3. Finally, check whether the code calling the model contains any errors, such as missing model names or critical elements (e.g., prompts). Please note that it is essential to carefully verify whether the code calling the LLM API is correct. If parameters such as api_key, model name, or stop tokens are missing, or if these parameters exist but have empty or None values, they should be considered incorrect. Additionally, please note to ignore errors where a variable is not defined within the provided code snippet. Below are some common standard calling patterns to pay special attention to: OpenAI API Call: from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "write a haiku about ai"}
    ]
)
Claude API Call:
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="",
)
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
Gemini API Call:
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
Carefully identify whether the LLM API calls in the code are correct and check for missing required parameters or incorrect initialization values for these parameters. Please return the result in the following JSON format: {
  "Name of the function": "Function name (if none, return None)",
  "Is there a function that calls an LLM?": "yes or no?",
  "Method of calling the model": "API or Local?",
  "Does the LLM initialization contain any defects?": "yes or no?",
  "Defect information": "Output the defect information when initializing the model (if none, return None)"
}\n """
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




def check_ETE_defect(data):
    data = str(data)
    if data == '' or not data:
        AMF_results = None
        return AMF_results, None,None
    
    if isinstance(data, str):
        try:
            clean_data = re.search(r'{[\s\S]*}', data)
            if clean_data is None:
                return None, None, None  
            json_data = clean_data.group()
            data = json.loads(json_data)  
        except json.JSONDecodeError as e:
            return "Invalid JSON format", None ,None 

    if not isinstance(data, dict):
        return "Input is not a dictionary", None,None

    for key, value in data.items():
        if value == "None" or value is None or not value:  
            return f"{key} is None", None, None
    
    defect_detail = data.get("Defect information")
    if data.get("Name of the function") == "None":
        return "Unknow ETE defects, function name is empty.", None, defect_detail
    
    else:
        if data.get("Is there a function that calls an LLM?") == "yes":
            if data.get("Does the LLM initialization contain any defects?") == "yes":
                if data.get("Method of calling the model") == "API":
                    ETE_results = True
                    return ETE_results, True,defect_detail
                else:
                    ETE_results = True
                    return ETE_results, False,defect_detail
            else:
                if data.get("Method of calling the model") == "API":
                    ETE_results = False
                    return ETE_results, True,defect_detail
                else:
                    ETE_results = False
                    return ETE_results, False,defect_detail
        else:
            return None, None, None


    

def handle_case(case, agent_llm_value, class_code_dict, class_name, folder_path, log_messages):
    with open(result_file, 'a') as f:
        if case == 'Case 1':
            f.write("LLM initialization parameter is None or there is a parsing error. \n\n")
        elif case == 'Case 3':
            f.write(f"Exist LARL Defect. The type of LLM initialization parameter in {class_name} is None. \n\n")
        elif case == 'Case 2':
            for agent_llm_value_item in agent_llm_value:
                agent_llm_value_location_query = f"""
                importCode(inputPath="{importProject}",projectName="{projectName}")
                cpg.typeDecl.name("{agent_llm_value_item}").l
                """
                process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(agent_llm_value_location_query.encode())                       
                location_results = stdout.decode('utf-8')
                location_cleaned = clean_string(location_results)
                class_info_list = extract_class_info(location_cleaned)
                class_code_dict,log_messages_1 = extract_class_code(folder_path, class_info_list)
                
                file_content = read_file_to_string(log_messages_1)
                check_results = process_content(file_content)
                extract_check_results = extract_relevant_json(check_results)
                result_classes = parse_classes(check_results)

                if "LLM Classes" in result_classes:
                    handle_llm_classes_logic(result_classes, class_code_dict, folder_path, log_messages_1)
                elif "Agent Classes" in result_classes:
                    handle_agent_classes_logic(result_classes, class_code_dict, folder_path, log_messages_1)
                else:
                    f.write(f"LLM initialization parameter is None or there is a parsing error.\n\n")
        else:
            f.write(f"Unknown case: {case}")





def handle_llm_classes_logic(result_classes, class_code_dict, folder_path, output_file):
    class_names = result_classes["LLM Classes"]
    with open(result_file, 'a') as f:
        if isinstance(class_names, list) and len(class_names) > 0:
            for class_name in class_names:
                try:
                    llm_result = handle_llm_classes(class_name, class_code_dict)
                     
                    check_llm_chat_result = llm_check_chat(llm_result)
                    LARL_results,call_way,LARL_results_extend, attri_extend= check_LARL_defect(check_llm_chat_result)
                    file_paths = get_file_paths_by_class_name(class_code_dict, class_name)
                    if LARL_results is False:
                        if call_way == "API":
                            if LARL_results_extend == False:
                                f.write(f"No LARL Defect for class {class_name}, {check_llm_chat_result}. However, please be mindful of the API balance. \n")
                                f.write(f"And All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n\n")
                            else:
                                f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths}. And please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Please be mindful of the API balance. \n\n")
                        else:
                            if LARL_results_extend == False:
                                f.write(f"No LARL Defect for class {class_name}, {check_llm_chat_result}.\n")
                                f.write(f"And All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n\n")
                            else:
                                f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths}. And please pay attention to the initial values of the input parameters. \n")

                    elif LARL_results is True:
                        if call_way == "API":
                            if LARL_results_extend == False:
                                f.write(f"All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Exist LARL Defect. Class {class_name} in {file_paths} has LARL defect, detail: {check_llm_chat_result}. And please be mindful of the API balance.\n\n")
                            else:
                                f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths}. And please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Exist LARL Defect. Class {class_name} in {file_paths} has LARL defect, detail: {check_llm_chat_result}. And please be mindful of the API balance.\n\n")
                        else:
                            if LARL_results_extend == False:
                                f.write(f"All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Exist LARL Defect. Class {class_name} in {file_paths} has LARL defect, detail: {check_llm_chat_result}. \n\n")
                            else:
                                f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths}. And please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Exist LARL Defect. Class {class_name} in {file_paths} has LARL defect, detail: {check_llm_chat_result}. \n\n")
                    else:
                        if call_way == "API":
                            if LARL_results_extend == False:
                                f.write(f"All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Unknown LARL result for class {class_name}: {LARL_results}, {check_llm_chat_result}. And please be mindful of the API balance.\n\n")
                            else:
                                f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths}. And please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Unknown LARL result for class {class_name}: {LARL_results}, {check_llm_chat_result}. And please be mindful of the API balance.\n\n")
                        else:
                            if LARL_results_extend == False:
                                f.write(f"All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Unknown LARL result for class {class_name}: {LARL_results}, {check_llm_chat_result}. \n\n")
                            else:
                                f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths}. And please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Unknown LARL result for class {class_name}: {LARL_results}, {check_llm_chat_result}. \n\n")
                    
                except Exception as e:
                    f.write(f"Error processing class {class_name}: {str(e)}\n\n")
        else:
            f.write(f"No valid LLM Classes found in {result_classes}\n\n")



def handle_agent_classes_logic(result_classes, class_code_dict, folder_path, log_messages):
    agent_classes = result_classes["Agent Classes"]
    with open(result_file, 'a') as f:
        for class_name in agent_classes:
            if class_name in class_code_dict:
                for file_path, class_code in class_code_dict[class_name].items():
                    agent_llm_result = handle_llm_classes(class_name, class_code_dict)
                    check_agent_have_llm_res = llm_check_initialization(agent_llm_result)
                    agent_have_llm_res_parse,call_way,LARL_results_extend, attri_extend = check_LARL_defect(check_agent_have_llm_res)
                
                    # target_code = extract_function_code(class_code_dict,class_name,fun_name)
                    # if target_code is not None:
                    #     LARL_results_extend,attri_extend = attr_flow_in(class_name,fun_name,target_code)
                    file_paths = get_file_paths_by_class_name(class_code_dict, class_name)
                    if agent_have_llm_res_parse is None or not agent_have_llm_res_parse:
                        attributes = extract_class_attributes(class_name, class_code)
                        extract_llm_from_gpt = llm_check_agent_attr(attributes)
                        extract_llm_info_from_agent_class = extract_agent_info(extract_llm_from_gpt)
                        case, agent_llm_value = evaluate_agent_llm_data(extract_llm_info_from_agent_class)
                        handle_case(case, agent_llm_value, class_code_dict, class_name, folder_path, log_messages)
                    elif agent_have_llm_res_parse is False:
                        if call_way == "API":
                            if LARL_results_extend == False:
                                f.write(f"No LARL Defect for class {class_name}, {check_agent_have_llm_res}. However, please be mindful of the API balance. \n")
                                f.write(f"And All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n\n")
                            else:
                                f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths} not have initial values. And please pay attention to the initial values of the input parameters. \n")
                                f.write(f"Please be mindful of the API balance. \n\n")
                        else:
                            if LARL_results_extend == False:
                                f.write(f"No LARL Defect for class {class_name}, {check_agent_have_llm_res}.\n")
                                f.write(f"And All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n\n")
                            else:
                                f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths} not have initial values. And please pay attention to the initial values of the input parameters. \n")

                        
                    elif agent_have_llm_res_parse is True:
                            if call_way == "API":
                                if LARL_results_extend == False:
                                    f.write(f"All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n")
                                    f.write(f"Exist LARL Defect. Class {class_name} in {file_paths} has LARL defect, detail: {check_agent_have_llm_res}. And please be mindful of the API balance.\n\n")
                                else:
                                    f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths} not have initial values. And please pay attention to the initial values of the input parameters. \n")
                                    f.write(f"Exist LARL Defect. Class {class_name} in {file_paths} has LARL defect, detail: {check_agent_have_llm_res}. And please be mindful of the API balance.\n\n")
                            else:
                                if LARL_results_extend == False:
                                    f.write(f"All necessary variables have initial values. Please pay attention to the initial values of the input parameters. \n")
                                    f.write(f"Exist LARL Defect. Class {class_name} in {file_paths} has LARL defect, detail: {check_agent_have_llm_res}. \n\n")
                                else:
                                    f.write(f"Exist LARL Defect: {attri_extend}. The {class_name} of {file_paths} not have initial values. And please pay attention to the initial values of the input parameters. \n")
                                    f.write(f"Exist LARL Defect. Class {class_name} in {file_paths} has LARL defect, detail: {check_agent_have_llm_res}. \n\n")




def handle_no_classes_logic(result_classes, folder_path, log_messages, class_code_dict):
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
                        res_1 = llm_check_method_attri(code)
                        ETE_results, API_status,defect_detail = check_ETE_defect(res_1)
                        if ETE_results is False:
                            if API_status is True:
                                f.write(f"No LARL Defect for method {func_name} of {path}, detail: {defect_detail}.\n ")
                                f.write(f"However, please be mindful of the API balance.\n")
                            else:
                                f.write(f"No LARL Defect for method {func_name} of {path}, detail: {defect_detail}.\n ")
                                f.write(f"Local LLM.\n")
                        elif ETE_results is True:
                            if API_status is True:
                                f.write(f"Exist LARL Defect for method {func_name} of {path}, detail: {defect_detail}.\n ")
                                f.write(f"However, please be mindful of the API balance.\n")
                            else:
                                f.write(f"Exist LARL Defect for method {func_name} of {path}, detail: {defect_detail}.\n ")
                                f.write(f"Local LLM.\n")
                        else:
                            f.write(f"Unknown LARL result for method {func_name}: {defect_detail}. \n")
                    else:
                        path = func_data["path"]
                        f.write(f"Unknown LARL result, No LLM method of {path}.")
    elif "Agent Method" in res_method:
        with open(result_file, 'a') as f:
            function_names = res_method.get("Agent Method", [])
            for func_name in function_names:
                for func_data in extracted_data:
                    if func_data["name"] == func_name:
                        path = func_data["path"]
                        code = func_data["code"]
                        #print(f"Found function '{func_name}' in file: {path}")
                        # Call the llm_check_attri function with the extracted code
                        res_1 = llm_check_method_attri(code)
                        ETE_results, API_status,defect_detail = check_ETE_defect(res_1)
                        if ETE_results is False:
                            if API_status is True:
                                f.write(f"No LARL Defect for method {func_name} of {path}, detail: {defect_detail}. \n")
                                f.write(f"However, please be mindful of the API balance.\n")
                            else:
                                f.write(f"No LARL Defect for method {func_name} of {path}, detail: {defect_detail}. \n")
                                f.write(f"Local LLM.\n")
                        elif ETE_results is True:
                            if API_status is True:
                                f.write(f"Exist LARL Defect for method {func_name} of {path}, detail: {defect_detail}. \n")
                                f.write(f"However, please be mindful of the API balance.\n")
                            else:
                                f.write(f"Exist LARL Defect for method {func_name} of {path}, detail: {defect_detail}. \n")
                                f.write(f"Local LLM.\n")
                        else:
                            f.write(f"Unknown LARL result for method {func_name}: {defect_detail}. \n")
                    else:
                        path = func_data["path"]
                        f.write(f"Unknown LARL result, No LLM method of {path}.\n")

    else:
        with open(result_file, 'a') as f:
            f.write(f"Unknown LARL result, No LLM attributes. Detail: {res_method}.\n")



def main(importProject, projectName):
    class_info = get_class_info_from_joern(importProject, projectName)
    class_code_location = extract_class_info(class_info)
    class_code_dict,log_messages = extract_class_code(importProject, class_code_location)
    check_results = process_content("".join(log_messages))
    extract_check_results = extract_relevant_json(check_results)
    result_classes = parse_classes(check_results)


    if "LLM Classes" in result_classes:
        handle_llm_classes_logic(result_classes, class_code_dict, importProject, log_messages)
    
    elif "Agent Classes" in result_classes:
        handle_agent_classes_logic(result_classes, class_code_dict, importProject, log_messages)
    else:
        all_node_str = get_all_class(importProject, projectName)
        all_method_str = get_all_method(importProject, projectName)
        all_node = extract_valid(all_node_str)
        all_method = extract_valid(all_method_str)
        all_class = remove_elements(all_node,all_method)
        all_class_order = reorder_class_list(all_class)
        class_detail = get_classes_detail(importProject, projectName, all_class_order)
        class_code_location_extend = extract_class_info(class_detail)
        class_code_dict_extend,olog_messages_extend = extract_class_code(importProject, class_code_location_extend)
        check_results_extend = process_content("".join(olog_messages_extend))
        extract_check_results = extract_relevant_json(check_results_extend)
        result_classes_extend = parse_classes(check_results_extend)
        if "LLM Classes" in result_classes_extend:
            handle_llm_classes_logic(result_classes_extend, class_code_dict_extend, importProject, log_messages)
    
        elif "Agent Classes" in result_classes_extend:
            handle_agent_classes_logic(result_classes_extend, class_code_dict_extend, importProject, log_messages)
        else:
            handle_no_classes_logic(result_classes_extend, importProject, log_messages, class_code_dict_extend)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_code_new.py <import_project> <project_name>")
        sys.exit(1)

    global importProject
    global projectName

    importProject = sys.argv[1]
    projectName = sys.argv[2]
    
    main(importProject, projectName)