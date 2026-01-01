import datetime
import re
from typing import Dict, List, Tuple

from pydantic import BaseModel

from llm_agents.llm import ChatLLM
from llm_agents.tools.base import ToolInterface
from llm_agents.tools.python_repl import PythonREPLTool


FINAL_ANSWER_TOKEN = "Final Answer:"
OBSERVATION_TOKEN = "Observation:"
THOUGHT_TOKEN = "Thought:"
PARSE_ERROR_TOOL = "__parse_error__"

PROMPT_TEMPLATE = """Today is {today} and you can use tools to get new information. Answer the question as best as you can using the following tools:

{tool_description}

Use the following format (STRICT):

Question: the input question you must answer
Thought: comment on what you want to do next
Action: the action to take, exactly one element of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Repeat Thought/Action/Action Input/Observation as needed until you are sure of the answer.
Thought: I now know the final answer
Final Answer: your final answer to the original input question

Begin!

Question: {question}
{previous_responses}
Thought:
"""


class Agent(BaseModel):
    llm: ChatLLM
    tools: List[ToolInterface]
    prompt_template: str = PROMPT_TEMPLATE
    max_loops: int = 15

    # Stop tokens to reduce hallucinated observations
    stop_pattern: List[str] = [
        f"\n{OBSERVATION_TOKEN}",
        f"\n\t{OBSERVATION_TOKEN}",
        f"\n {OBSERVATION_TOKEN}",
    ]

    # Controls for robustness / MNFT
    max_tool_input_chars: int = 4000
    max_observation_chars: int = 4000

    @property
    def tool_description(self) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

    @property
    def tool_names(self) -> str:
        return ",".join([tool.name for tool in self.tools])

    @property
    def tool_by_names(self) -> Dict[str, ToolInterface]:
        # 防 IETI：同名工具会被覆盖 -> 直接报错提醒改名
        tools: Dict[str, ToolInterface] = {}
        for tool in self.tools:
            if tool.name in tools:
                raise ValueError(f"Duplicate tool name detected: {tool.name!r}. Rename tools to unique names.")
            tools[tool.name] = tool
        return tools

    def _safe_tool_use(self, tool: ToolInterface, tool_input: str) -> str:
        # 输入检查
        if tool_input is None:
            return "ToolError: tool_input is None"
        if not isinstance(tool_input, str):
            tool_input = str(tool_input)

        tool_input = tool_input.strip()
        if not tool_input:
            return "ToolError: empty tool_input"

        # 防止超长输入拖垮工具/网络请求
        if len(tool_input) > self.max_tool_input_chars:
            tool_input = tool_input[: self.max_tool_input_chars] + "...(truncated)"

        try:
            out = tool.use(tool_input)
            if out is None:
                return "ToolError: tool returned None"
            if not isinstance(out, str):
                out = str(out)

            # 防止 observation 撑爆上下文（MNFT）
            if len(out) > self.max_observation_chars:
                out = out[: self.max_observation_chars] + "...(truncated)"
            return out

        except Exception as e:
            return f"ToolError: {type(e).__name__}: {e}"

    def run(self, question: str):
        previous_responses: List[str] = []
        num_loops = 0

        prompt = self.prompt_template.format(
            today=datetime.date.today(),
            tool_description=self.tool_description,
            tool_names=self.tool_names,
            question=question,
            previous_responses="{previous_responses}",
        )

        print(prompt.format(previous_responses=""))

        while num_loops < self.max_loops:
            num_loops += 1
            curr_prompt = prompt.format(previous_responses="\n".join(previous_responses))

            generated, tool, tool_input = self.decide_next_action(curr_prompt)

            if tool == "Final Answer":
                return tool_input

            if tool == PARSE_ERROR_TOOL:
                tool_result = (
                    "ParseError: could not parse tool call. Please follow the required format. "
                    f"Raw output: {tool_input!r}"
                )
                step = f"{generated}\n{OBSERVATION_TOKEN} {tool_result}"
                print(step + f"\n{THOUGHT_TOKEN}")
                previous_responses.append(step)

                open('agent_log.txt','a',encoding='utf-8').write(step+'\n')  # BUG
                continue

            tool_map = self.tool_by_names()
            if tool not in tool_map:
                tool_result = f"ToolError: unknown tool {tool!r}. Available: {list(tool_map.keys())}"
                step = f"{generated}\n{OBSERVATION_TOKEN} {tool_result}"
                print(step + f"\n{THOUGHT_TOKEN}")
                previous_responses.append(step)
                continue

            tool_result = self._safe_tool_use(tool_map[tool], tool_input)
            step = f"{generated}\n{OBSERVATION_TOKEN} {tool_result}"
            print(step + f"\n{THOUGHT_TOKEN}")
            previous_responses.append(step)

        raise RuntimeError(f"Exceeded max_loops={self.max_loops} without reaching Final Answer.")

    def decide_next_action(self, prompt: str):
        # 第一次生成
        generated = self.llm.generate(prompt, stop=self.stop_pattern)
        tool, tool_input = self._parse(generated)

        # 解析失败：让 LLM 严格重写一次（最小改动的“自修复”容错）
        if tool == PARSE_ERROR_TOOL:
            repair_prompt = (
                prompt
                + "\n\nYour previous output was not parsable. "
                  "Rewrite it STRICTLY in the required format with lines:\n"
                  "Thought: ...\nAction: <one tool name>\nAction Input: ...\n"
                  "OR\nThought: ...\nFinal Answer: ...\n"
                  "Return only those lines."
            )
            generated2 = self.llm.generate(repair_prompt, stop=self.stop_pattern)
            tool2, tool_input2 = self._parse(generated2)
            if tool2 != PARSE_ERROR_TOOL:
                return generated2, tool2, tool_input2

        return generated, tool, tool_input

    def _parse(self, generated: str) -> Tuple[str, str]:
        if not isinstance(generated, str):
            return PARSE_ERROR_TOOL, f"non-string output: {type(generated).__name__}"

        if FINAL_ANSWER_TOKEN in generated:
            return "Final Answer", generated.split(FINAL_ANSWER_TOKEN)[-1].strip()

        regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, generated, re.DOTALL)
        if not match:
            return PARSE_ERROR_TOOL, generated

        tool = match.group(1).strip()
        tool_input = match.group(2)
        return tool, tool_input.strip(" ").strip('"')


if __name__ == "__main__":
    agent = Agent(llm=ChatLLM(), tools=[PythonREPLTool()])
    result = agent.run("What is 7 * 9 - 34 in Python?")
    print(f"Final answer is {result}")
