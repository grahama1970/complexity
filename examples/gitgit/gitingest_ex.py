# Synchronous usage
from gitingest import ingest

# summary, tree, content = ingest("repos/gitingest")

# or from URL
summary, tree, content = ingest(
    "https://github.com/cyclotruc/gitingest",
    include_patterns={"*.md"},
    output="repos/gitingest"
    )
print("Summary:", summary)
print("Tree:", tree)