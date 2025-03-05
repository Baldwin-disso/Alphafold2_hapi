#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPOS_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"



# Define the path to the parent directory
parent_dir=${REPOS_DIR}/outputs


# Check if the parent directory exists
if [ ! -d "$parent_dir" ]; then
  echo "Error: Directory '$parent_dir' does not exist."
  exit 1
fi

# Loop through the immediate subdirectories of the parent directory
for folder in "$parent_dir"/*; do
  # Skip if not a directory
  if [ ! -d "$folder" ]; then
    continue
  fi

  echo "Processing '$folder'"

  # Preserve the "msas" subdirectory if it exists
  msas_dir="$folder/msas"

  # Find all paths to exclude (msas and all its content)
  if [ -d "$msas_dir" ]; then
    exclude_paths=$(find "$msas_dir" -type d -print)
    exclude_files=$(find "$msas_dir" -type f -print)
  else
    exclude_paths=""
    exclude_files=""
  fi

  # Remove all files and directories in the current folder except msas and its content
  find "$folder" -mindepth 1 | while read -r path; do
    if [[ "$exclude_paths" == *"$path"* ]] || [[ "$exclude_files" == *"$path"* ]]; then
      # Skip paths that are part of msas
      continue
    fi
    echo "Removing: $path"
    rm -rf "$path"
  done
done

echo "Cleanup complete."
