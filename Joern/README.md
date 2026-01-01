# Joern — Minimal README

Joern is an open-source code analysis platform for analyzing source code, bytecode, and binary executables. It generates **Code Property Graphs (CPGs)** and lets you mine them using a Scala-based query language (CPGQL), commonly used for vulnerability discovery and static program analysis.

---

## Prerequisites
- A working Java runtime (Joern runs on the JVM).
- Enough memory for large projects (you can pass JVM flags like `-J-Xmx30G` when needed).

---

## Install (pre-built binaries)
The official docs recommend installing via the release installer script:

```bash
mkdir joern && cd joern   # optional
curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" -o joern-install.sh
chmod u+x joern-install.sh
./joern-install.sh --interactive
```

By default, Joern is installed under `~/bin/joern`.

**Test the installation** (path may vary depending on where you installed it):

```bash
cd <path_to_joern>/joern/joern-cli
./joern
```

You should see the Joern banner and a `joern>` prompt.

---

## Quickstart (interactive)
Start the interactive shell:

```bash
joern
```

Import a codebase as a project:

```scala
importCode(inputPath = "./path/to/code", projectName = "my-project")
```

Notes:
- `importCode` creates a project directory, generates a CPG, loads it into memory, and prepares overlays.
- If `importCode` returns `None`, the input path is likely incorrect.

---

## Common queries (copy & run)
List all methods:

```scala
cpg.method.l
```

List method names:

```scala
cpg.method.name.l
```

Find call sites by callee name (example: `strcpy`):

```scala
cpg.call.name("strcpy").code.l
```

Find call sites *into* a method (example: `exit`) and print the call code:

```scala
cpg.method.name("exit").callIn.code.l
```

Find calls *out of* `main` and list their names:

```scala
cpg.method.name("main").callOut.name.l
```

---

## Export / visualize graphs
Joern provides interactive plotting and the `joern-export` utility for exporting graph representations.

Examples (interactive plotting varies by Joern version and plugins):
```scala
// Explore available plotting helpers via TAB-completion, e.g.:
// cpg.method.name("foo").plotDotAst
```

For batch export, see `joern-export` in the official docs.

---

## Troubleshooting
### Import fails or runs out of memory
For large codebases, increase JVM heap size:

```bash
joern -J-Xmx30G
```

If `importCode` suggests running the frontend directly (e.g., `c2cpg.sh`), follow the guidance printed by Joern and then import the generated CPG with `importCpg(...)`.

---

## References
- Joern Documentation (Overview): https://docs.joern.io/
- Quickstart: https://docs.joern.io/quickstart/
- GitHub Releases: https://github.com/joernio/joern/releases/
