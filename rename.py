import os

replacements = {
    "github.com/sakshamgoswami/synapse-cache": "github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine",
    "github.com/sakshamgoswami/nearby": "github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine",
    "synapse-cache": "nearby",
}

for root, dirs, files in os.walk("."):
    if ".git" in root or ".venv" in root or "__pycache__" in root:
        continue
    for file in files:
        # Avoid changing rename.py itself
        if file == "rename.py":
            continue
        if file.endswith((".go", ".mod", ".md", ".yml", "Dockerfile", ".py", ".txt", ".json")) or "Dockerfile" in file:
            filepath = os.path.join(root, file)
            with open(filepath, "r") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError:
                    continue
            
            new_content = content
            # Apply replacements in order
            # First module URLs
            new_content = new_content.replace("github.com/sakshamgoswami/synapse-cache", "github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine")
            new_content = new_content.replace("github.com/sakshamgoswami/nearby", "github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine")
            
            # Then the general project name
            new_content = new_content.replace("synapse-cache", "nearby")
            
            if new_content != content:
                with open(filepath, "w") as f:
                    f.write(new_content)
                print(f"Updated {filepath}")
