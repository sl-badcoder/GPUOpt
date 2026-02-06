#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <path/to/filename.cu>"
    exit 1
fi

SOURCE_FILE="$1"

# Transform path/to/file.cu -> path_to_file
WITHOUT_EXT="${SOURCE_FILE%.cu}"
FLATTENED_NAME="${WITHOUT_EXT//\//_}"

TEMP_BINARY="./temp_binary_$(date +%s)" # Unique temp name to avoid collisions

# Date and Time formatting
TIMESTAMP=$(date +%H:%M:%S)
DATE=$(date +%d_%m_%Y)
PROFILING_DIR="results/profiling/${DATE}"

mkdir -p "$PROFILING_DIR"
OUTPUT_PATH="${PROFILING_DIR}/profile_${FLATTENED_NAME}_${TIMESTAMP}"

echo "Compiling $SOURCE_FILE..."
nvcc "$SOURCE_FILE" -o "$TEMP_BINARY"

if [ $? -ne 0 ]; then
    echo "Compilation failed! Profile aborted."
    exit 1
fi

echo "Starting profile for $SOURCE_FILE..."
nsys profile \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true \
    --force-overwrite true \
    -o "$OUTPUT_PATH" \
    "$TEMP_BINARY"

# Cleanup
rm "$TEMP_BINARY"

echo "------------------------------------------------"
echo "Done!"
echo "Files created in $PROFILING_DIR:"
echo " - profile_${FLATTENED_NAME}_${TIMESTAMP}.nsys-rep"
echo " - profile_${FLATTENED_NAME}_${TIMESTAMP}.sqlite"