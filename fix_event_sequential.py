"""Fix race condition: make vault event handler sequential."""
with open("api/main.py") as f:
    lines = f.readlines()

start = None
end = None
for i, l in enumerate(lines):
    if "def _on_vault_event" in l:
        start = i
    if start is not None and i > start and "def _handle_vault_event_sync" in l:
        end = i
        break

if start is not None and end is not None:
    # Replace the function body
    new_lines = [
        "def _on_vault_event(event: VaultEvent) -> None:\n",
        '    """Handle vault file changes sequentially to prevent race conditions."""\n',
        "    _handle_vault_event_sync(event)\n",
        "\n",
        "\n",
    ]
    lines[start:end] = new_lines
    with open("api/main.py", "w") as f:
        f.writelines(lines)
    print(f"Replaced lines {start+1} to {end}")
else:
    print(f"Could not find function boundaries: start={start}, end={end}")
