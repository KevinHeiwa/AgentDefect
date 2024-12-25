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


Model = "gpt"
result_file = ""


def get_class_info_from_joern():
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}");
    cpg.typeDecl.name.l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean


def get_method_info_from_joern():
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}");
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
    return [item for item in list1 if item not in list2]
    #return [item for item in list1 if item not in list2 and ('tool' in item.lower())]



def query_class_inheritance(classes):
    inheritance_dict = {}
    for class_name in classes:
        joern_query = f"""
        importCode(inputPath="{importProject}", projectName="{projectName}");
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

        # print(class_name)
    
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
        ## print(f"Error: The file '{file_in}' was not found.")
        return None
    except Exception as e:
        ## print(f"An error occurred: {e}")
        return None
    


def query_class_details(class_name):
    
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}");
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
        ## print("No JSON-like structure found in the input string.")
        return None
    json_str = match.group()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        ## print("Failed to parse JSON string.")
        return None
    is_related_to_tool = data.get("Is it related to Tool?")
    is_initial_class_of_tool = data.get("Is it the initial class of Tool?")
    if is_related_to_tool == "yes" and is_initial_class_of_tool == "yes":
        return data.get("Class Name")
    return None



def process_layers(layers):
    for level in layers:
        ## print(f"Processing level: {level}")
        for class_name in level:
            details  = query_class_details(class_name)
            
            if details:
                filename, line_number = details[0]                
                class_code = extract_class_code(filename, class_name, line_number)               
                llm_check_res = llm_check_tool_initialization_class(class_code)
                # # print(f"the res is:{llm_check_res}")
                find_res = extract_tool_class_info(llm_check_res)
                if find_res is not None:
                    return class_name 
    return None




def get_classextend_info_joern(class_name):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}");
    cpg.typeDecl.filter(_.inheritsFromTypeFullName.exists(_.matches(".*{class_name}.*"))).l
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    ## print(result_clean)
    return result_clean


def extract_type_decl_info(input_str):
    name_pattern = r'\bname\s*=\s*"([^"]+)"'
    filename_pattern = r'\bfilename\s*=\s*"([^"]+)"'
    line_number_pattern = r'\blineNumber\s*=\s*Some\(value\s*=\s*(\d+)\)'

    type_decl_blocks = re.split(r'TypeDecl\(', input_str)[1:]  
    
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
        # print(f"SyntaxError in function {function_name}: {e}")
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
                        ## print(f"SyntaxError in line {i}: {e}")
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
        ## print(f"Error: The file '{file_in}' was not found.")
        return None
    except Exception as e:
        ## print(f"An error occurred: {e}")
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
        # print("No JSON-like structure found in the input string.")
        return None
    json_str = match.group()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # print("Failed to parse JSON string.")
        return None

    return data
    


class FunctionReturnChecker(ast.NodeVisitor):
    def __init__(self):
        self.functions_without_return = []
        self.final_return_values = []
    
    def visit_FunctionDef(self, node):
        # Check if the function has any return statements
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        if not has_return:
            self.functions_without_return.append(node.name)
        else:
            # Collect return values
            for n in ast.walk(node):
                if isinstance(n, ast.Return):
                    if n.value:
                        try:
                            # Use ast.unparse for more readable return values
                            return_value = ast.unparse(n.value) if hasattr(ast, 'unparse') else ast.dump(n.value)
                            self.final_return_values.append(return_value)
                        except Exception as e:
                            self.final_return_values.append(f"Unparseable return value: {str(e)}")
        self.generic_visit(node)



def check_class_and_functions_return(class_code: str) -> tuple[bool, str]:
    """
    Check if all functions in the class have return values and return the final return value if all are valid.

    :param class_code: The class code to be checked.
    :return: A tuple containing a boolean and a message or return value.
    """
    try:
        # Parse the class code into an AST
        tree = ast.parse(class_code)
        
        # Initialize the function return checker
        checker = FunctionReturnChecker()
        
        # Visit the parsed tree
        checker.visit(tree)

        # Check if there are any functions without return values
        if checker.functions_without_return:
            return False, f"Function(s) without return: {', '.join(checker.functions_without_return)}"
        
        # If all functions have return values, return True and the final return value
        final_return = ", ".join(checker.final_return_values) if checker.final_return_values else "No explicit return values"
        return True, final_return
    except Exception as e:
        return False, f"Exist empty function, detail: {str(e)}"
    


def get_stop_info_from_joern():
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}");
    cpg.assignment.code(".*stop.*").filter(a => a.source.code.matches(".*\\\\[.*\\\\]")).map(a => {{ (a.target.code, a.source.code) }}).foreach(result => println(result._1 + " : " + result._2));
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean



# def find_llm_or_agent_calls(file_path):
#     stop_words = set()  
#     with open(file_path, "r", encoding="utf-8") as source:
#         tree = ast.parse(source.read())
#         for node in ast.walk(tree):
#             if isinstance(node, (ast.FunctionDef, ast.ClassDef)):  
#                 for child in ast.walk(node):
                
#                     if isinstance(child, ast.keyword) and child.arg == "stop":
#                         if isinstance(child.value, ast.List):
#                             stop_words.update([elt.s for elt in child.value.elts if isinstance(elt, ast.Str)])

#                     if isinstance(child, ast.Assign): 
#                         for target in child.targets:
#                             if isinstance(target, ast.Name) and target.id == "stop":
#                                 if isinstance(child.value, ast.List):
#                                     stop_words.update(
#                                         [elt.s for elt in child.value.elts if isinstance(elt, ast.Str)]
#                                     )
#     return stop_words

def find_llm_or_agent_calls(file_path):
    def extract_stop_words(value):
        if isinstance(value, ast.List):
            return {elt.value for elt in value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}
        return set()
    
    stop_words = set()  
    stop_names = {"stop", "stop_words", "stop_tokens"}  
    
    with open(file_path, "r", encoding="utf-8") as source:
        tree = ast.parse(source.read())
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):  
                for child in ast.walk(node):
                
                    if isinstance(child, ast.keyword) and child.arg in stop_names:
                        stop_words.update(extract_stop_words(child.value))
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id in stop_names:
                                stop_words.update(extract_stop_words(child.value))

    return stop_words


def extract_stop_words_from_project(project_path):
    all_stop_words = set()  
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"): 
                file_path = os.path.join(root, file)
                stop_words = find_llm_or_agent_calls(file_path)
                all_stop_words.update(stop_words) 
    return sorted(all_stop_words) 


def extract_value_from_result(result):
    match = re.search(r'\"(.*?)\"', result)
    if match:
        return match.group(1)  
    return result  
query_cache = {}


def extract_and_query(string,depth=0):
    if depth > MAX_RECURSION_DEPTH:
        return string  
    match = re.search(r"\{(.*?)\}", string)
    
    if not match:
        return string

    variable_name = match.group(1)
    if variable_name in query_cache:
        joern_result = query_cache[variable_name]
    else:
        joern_result = go_stop_info_from_joern(variable_name)
        query_cache[variable_name] = joern_result
    
    if not joern_result or "Not enough" in joern_result:
        joern_result = f"{{{variable_name}}}"
    else:
        extracted_value = extract_value_from_result(joern_result)
        if re.search(r"\{(.*?)\}", extracted_value):
            extracted_value = extract_and_query(extracted_value, depth + 1)
        
        joern_result = extracted_value
    updated_string = string.replace(f"{{{variable_name}}}", joern_result)
    if re.search(r"\{(.*?)\}", updated_string):
        return extract_and_query(updated_string, depth + 1)
    return updated_string



def go_stop_info_from_joern(token):
    joern_query = f"""
    importCode(inputPath="{importProject}",projectName="{projectName}");
    cpg.identifier.code("{token}").map(ident => ident.name + " : " + ident.astParent.code).foreach(println);
    """
    process = subprocess.Popen(['joern'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(joern_query.encode())
    result = stdout.decode('utf-8')
    result_clean = clean_string(result)
    return result_clean


def combine_and_check(tool_name, tool_output, res2):
    tool_output_list = [item.strip().strip("'") for item in tool_output.split(',')]
    combined_list = []
    for output in tool_output_list:
        combined_list.append(f"{tool_name}_{output}")  
        combined_list.append(f"{tool_name} {output}") 
        combined_list.append(f"{tool_name}-{output}")
        combined_list.append(f"{tool_name}{output}")   
        combined_list.append(f"{output}_{tool_name}")  
        combined_list.append(f"{output} {tool_name}")  
        combined_list.append(f"{output}-{tool_name}") 
        combined_list.append(f"{output}{tool_name}")   
    
    for value in res2:
        if value in combined_list:
            return True
    return False
    

def extract_quoted_strings(input_str):
    pattern = r'\'(.*?)\''
    return re.findall(pattern, input_str)



def check_list_inclusion(list1, str1, str2,file_name,tool_code):
    with open(result_file, 'a') as f:
        list2 = extract_quoted_strings(str1)
        list3 = extract_quoted_strings(str2)
        result = []
        full_path = importProject + "\\" + file_name
        for item in list1:
            if any(item in elem for elem in list2) or any(item in elem for elem in list3):
                f.write(f"Exist TV Defect. The return value of tools {str1} and vocabulary {item} is consistent in {full_path}. Detail:{str2}.\n")
                result.append((item, True))  
            else:
                f.write(f"No TV Defect. The return value of tools {str1} and vocabulary {item} is non-consistent in {full_path}. Detail:{str2}.\n")
                result.append((item, False)) 
    
    return result



def count_list_elements(input_string):

    try:
       
        processed_string = re.sub(r"f'(.*?)'", r"'\1'", input_string)
        start_index = processed_string.index('[')
        end_index = processed_string.index(']')
        list_content = processed_string[start_index:end_index + 1]
        parsed_list = ast.literal_eval(list_content)
        return len(parsed_list)
    except (ValueError, SyntaxError) as e:
        return 0
    
MAX_RECURSION_DEPTH = 0



def format_tool_info(two_d_array):
    combined_results = []
    global MAX_RECURSION_DEPTH
    with open(result_file, 'a') as f:
        res2 = get_stop_info_from_joern()
        res_ast = extract_stop_words_from_project(importProject)
        num = count_list_elements(res2)
        MAX_RECURSION_DEPTH = num + num
        res3 = extract_and_query(res2)
        for sub_array in two_d_array:
            if len(sub_array) >= 2:
                tool_name = sub_array[0]
                tool_code = sub_array[-1]
                file_name = sub_array[1]
                result = f"[Tool Name]\n {tool_name}\n[Tool Code]\n{tool_code}\n\n"
                result_1 = check_class_and_functions_return(tool_code)
                result_flag, tool_output = result_1
                if result_flag:
                    pattern = r'\'(.*?)\''
                    res3_list = re.findall(pattern, res3)
                    res3_list.extend(res_ast)
                    if not res3_list:  
                        res3_list = ['None','</s>','<|endoftext|>','<eos>']
                        #f.write(f"Exist TV Defect. The tool {tool_name} in {file_name} no stop token.\n")
                    res_check = check_list_inclusion(res3_list,tool_name,tool_output,file_name,tool_code)
                    res4 = combine_and_check(tool_name,tool_output,res3_list)
                    if res4 == True:
                        f.write(f"Exist TV Defect, the tool {tool_name} in {file_name}. Detail:{tool_output},{res2}.\n\n")
                    else:
                        f.write(f"No TV Defect. The tool {tool_name} in {file_name}.\n")
    return combined_results



def prompt_TRE(content):
    system_message = """You are a software engineer proficient in multiple programming languages, including Python and Java. You can accurately perform program analysis based on specific requirements and return results accordingly. \n """
    analysis_message = """Below is the name and code for a tool. We need to analyze inconsistencies between the tool’s implementation, name, and description. Here’s what you need to do: First, analyze whether the class in this code contains variables for the tool’s name and description (such as name and description). If they exist, further determine whether these two variables are empty. Next, summarize the tool’s implementation code and consider whether there are any inconsistencies between the tool’s code implementation, name, and description. Finally, must return the result only in the following JSON format: {"Tool Name": "Please print the name of the Tool.", "Are there variables representing the tool's name and description in the code?": "yes or no?", "Please output the variables representing the tool's name and description": "Output variable names (if empty, return None)", "Please output the initialization values of the variables representing the tool's name": "Output the initialization values of the variables (if empty, return None)", "Please output the initialization values of the variables representing the tool's description": "Output the initialization values of the variables (if empty, return None)", "Are there inconsistencies between the tool's implementation and its name and description?": "yes or no?" } \n"""
    analysis_object = content
    combined_string = system_message + analysis_message  + analysis_object
    return combined_string



def llm_check_TRE(content):
    new_content = prompt_TRE(content)
    messages = [{'role': 'user','content': ''},]
    messages[0]['content'] = new_content
    response = client.chat.create(
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
    analysis_message = """Please analyze the following code as per the instructions. The code below contains some function source code from an Agent project. Typically, an Agent project includes various independent tools, such as a calculator or weather tool. Firstly, analyze these functions one by one and identify if each function is the implementation of a particular tool. If so, check if this code includes any variable specifying the tool’s name and another variable specifying its description. If there are variables specifying both the tool’s name and description, further analyze if there is any inconsistency between the tool’s name, description, and its actual implementation. For example, the tool name might be ‘calculator’ while the description refers to a tool for retrieving real-time weather. Another example might be an implementation for a data retrieval tool with the name ‘today’s news.’ Note that if the tool description is insufficient, such as providing only a partial description of the tool’s functionality or if the description statement is incomplete, this is also considered an inconsistency between the tool’s name, description, and actual code implementation. Please identify this part strictly. Please output only the result in the following JSON format, without additional commentary: {"method Name": "Please print the name of the method.", "is Tool Implementation Instance": "yes or no", "Tool Name Variable": "Please output the variable name (if none, output None)", "Tool Description Variable": "Please output the variable name (if none, output None)", "Tool Name Variable Initialization Value": "Output the initialization value of the tool name variable (if none, output None)", "Tool Description Variable Initialization Value": "Output the initialization value of the tool description variable (if none, output None)", "Is there an inconsistency between Tool Name, Description, and Implementation": "yes or no", "Detailed Description of the Issue": "Specifically describe the identified inconsistency"}\n """
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
            if all(key in json_obj for key in ["method Name", "is Tool Implementation Instance", "Tool Name Variable", "Tool Description Variable", "Tool Name Variable Initialization Value", "Tool Description Variable Initialization Value", "Is there an inconsistency between Tool Name, Description, and Implementation", "Detailed Description of the Issue"]):
                extracted_json_list.append(json.dumps(json_obj, indent=2))
        except json.JSONDecodeError:
            continue
    return '\n'.join(extracted_json_list)


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



def extract_tool_methods(data_str, function_list):

    json_blocks = re.findall(r'\{.*?\}', data_str, re.DOTALL)
    method_names = []

    for block in json_blocks:
        data = json.loads(block)
        if data.get("is Tool Implementation Instance") == "yes":
            method_names.append(data.get("method Name"))
    result = []
    for method_name in method_names:
        for func in function_list:
            if func["name"] == method_name:
                name = func["name"]
                path = func.get("path", None)
                code = func.get("code", None)
                match = re.search(r"def " + re.escape(method_name) + r"\(.*?\):", code)
                line_number = None
                if match:
                    line_number = code[:match.start()].count("\n") + 1
                result.append([name, path, line_number, code])
                break

    return result



#def main(importProject, projectName): 
def main(): 
    class_info = get_class_info_from_joern()
    method_info = get_method_info_from_joern()
    new = extract_valid_class_names(class_info)
    new1 = extract_valid_class_names(method_info)
    result = remove_elements(new, new1)
    res1 = query_class_inheritance(result)
    inheritance_tree = build_inheritance_tree(res1)
    level_class = level_order_traversal(inheritance_tree)
    res2 = process_layers(level_class)
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
            method_check_1 = llm_check_method(i)
            method_check += method_check_1
            method_check += '\n'
        No_res = extract_relevant_json_1(method_check)
        method_res = extract_tool_methods(No_res,extracted_data)
        useless_res = format_tool_info(method_res)

    else:
        res3 = get_classextend_info_joern(res2)
        res4 = extract_type_decl_info(res3)
        res5 = process_class_codes(res4)
        res6 = format_tool_info(res5)
        with open(result_file, 'a') as f:
            f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_code_new.py <import_project> <project_name>")
        sys.exit(1)

    global importProject
    global projectName

    importProject = sys.argv[1]
    projectName = sys.argv[2]
    
    main()