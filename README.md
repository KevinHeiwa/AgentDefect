# AgentDefects
This repository contains a preliminary version of Agentable, an CPG- and LLM-powered tool designed to detect code defects in Agents.

## Structure
|         **Folder**         |                        **Usage**                        |
|:--------------------------:|:-------------------------------------------------------:|
|            Data            |         Definition (Posts), AgentSet and AgentTest        |
| Detect |  Including detect code of Agent defect  |
|  Preprocess | Including preprocess code of Agent defect |

## Quick Start
```
pip install requirements -r
```
The operation of Agentable depends on Joern. You can visit the following websites for Joern support: https://github.com/joernio/joern and https://joern.io/.

```
python down.py
```
To use down.py to download Agent projects, make sure to specify the target folder where the projects should be saved.
```
python prescan.py
```
Since there are many irrelevant files in the Agent projects, such as documentation files, it is necessary to preprocess the Agent projects using the above command.
