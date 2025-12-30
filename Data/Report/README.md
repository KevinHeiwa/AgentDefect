# 📄 Report (How to Read)

This folder contains **8 detection reports**.  
Each report file corresponds to **one defect type** and lists the findings detected by our tool.  [oai_citation:0‡GitHub](https://github.com/KevinHeiwa/AgentDefect/blob/main/Data/Report/README.md)

## Files

| File | Defect |
| --- | --- |
| `ADAL.txt` | ADAL |
| `IETI.txt` | IETI |
| `LOPE.txt` | LOPE |
| `TRE.txt`  | TRE  |
| `ALS.txt`  | ALS  |
| `MNFT.txt` | MNFT |
| `LARD.txt` | LARD |
| `EPDD.txt` | EPDD |

--- 

## What is inside a report?

A report is a **plain-text list of findings**.  
Each finding tells you:

- **what** was detected (defect / warning)
- **where** it appears (file path, and sometimes class / function)
- **why** it is flagged (a short explanation)

## How to use the reports

1. Pick the defect type you care about (open the corresponding `.txt` file).
2. Read each finding and locate the code using the **file path + keywords** (class/function names).

## Quick tips

- If a path in the report does not exist on your machine, treat it as a **location hint** and search for the same filename/class in your repository.
- You can search within a report for keywords like class names, function names, or tool/LLM identifiers.