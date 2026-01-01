# AgentDefect — Data/Definition

This folder contains **5 CSV files** (`autogen.csv`, `flowise.csv`, `langchain.csv`, `llamaindex.csv`, `so.csv`).

Each CSV records **(1) a post link** and **(2) its defect type label**.

---

## Files

- `autogen.csv` — Posts collected from/for AutoGen
- `flowise.csv` — Posts collected from/for Flowise
- `langchain.csv` — Posts collected from/for LangChain
- `llamaindex.csv` — Posts collected from/for LlamaIndex
- `so.csv` — Stack Overflow posts

---

## CSV format

Each row corresponds to **one (post, defect-type)** pair:

- **Column 1**: `post_url` — the link to the post (e.g., a GitHub issue/discussion URL or a Stack Overflow question URL)
- **Column 2**: `defect_type` — one of the defect labels (e.g., `ADAL`, `IETI`, `LOPE`, `TRE`, `ALS`, `MNFT`, `LARD`, `EPDD`)

Example (illustrative):

```csv
https://example.com/post/123,ADAL
```

> Note: If the CSV includes a header row, treat it as column names; otherwise treat every row as data.

---

## Counting rule (multi-label posts)

Some posts may be labeled with **multiple defect types**.

In that case, the same `post_url` will appear **multiple times**, once per label.  
Therefore, **a multi-label post is counted multiple times** (one count for each label).

---

## Quick analysis (optional)

### Count labels in one CSV (Python / pandas)

```python
import pandas as pd

df = pd.read_csv("autogen.csv", header=None, names=["post_url", "defect_type"])
print(df["defect_type"].value_counts())
```

### Count how many posts have multiple labels

```python
import pandas as pd

df = pd.read_csv("autogen.csv", header=None, names=["post_url", "defect_type"])
multi = (df.groupby("post_url")["defect_type"].nunique() > 1).sum()
print("posts_with_multiple_labels =", int(multi))
```

---

## Notes

- These files are **definition/label mappings** (URL → defect type).  
- When aggregating statistics across files, remember that a post can be counted multiple times if it has multiple labels.
