#!/bin/bash

# Check if username and hostname arguments are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ssh_username> <remote_host>"
    echo "Example: $0 myuser server.example.com"
    exit 1
fi

# Configuration - Edit these variables
SSH_USERNAME="$1"
REMOTE_HOST_BASE="$2"
REMOTE_HOST="${SSH_USERNAME}@${REMOTE_HOST_BASE}"
REMOTE_FOLDER="~/working/models/results"
LOCAL_FOLDER="./external/results/metrics"
SSH_KEY_PATH=""  # Optional: path to SSH key, leave empty to use default
BASE_CSV_NAME="combined_metrics"  # Will be appended with timestamp

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Generate timestamp for filename
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
COMBINED_CSV_NAME="${BASE_CSV_NAME}_${TIMESTAMP}.csv"

echo -e "${YELLOW}Starting rsync and CSV combination process...${NC}"
echo -e "${YELLOW}Connecting to: $REMOTE_HOST${NC}"
echo -e "${YELLOW}Output file will be: $COMBINED_CSV_NAME${NC}"

# Create local folder if it doesn't exist
mkdir -p "$LOCAL_FOLDER"

# Build rsync command
RSYNC_CMD="rsync -avz --progress"
if [[ -n "$SSH_KEY_PATH" ]]; then
    RSYNC_CMD="$RSYNC_CMD -e 'ssh -i $SSH_KEY_PATH'"
fi
RSYNC_CMD="$RSYNC_CMD $REMOTE_HOST:$REMOTE_FOLDER/ $LOCAL_FOLDER/"

echo -e "${YELLOW}Syncing folder from remote server...${NC}"
echo "Command: $RSYNC_CMD"

# Execute rsync
eval $RSYNC_CMD

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Rsync completed successfully${NC}"
    
    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PYTHON_SCRIPT="$SCRIPT_DIR/combine-model-tests.py"
    
    # Check if Python script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        echo -e "${RED}✗ Python script not found: $PYTHON_SCRIPT${NC}"
        echo -e "${YELLOW}Looking for combine-model-tests.py in \
            current directory...${NC}"
        if [[ -f "./combine-test-results.py" ]]; then
            PYTHON_SCRIPT="./combine-model-tests.py"
            echo -e "${GREEN}✓ Found in current directory${NC}"
        else
            echo -e "${RED}✗ combine-model-tests.py not found${NC}"
            exit 1
        fi
    fi
    # Run Python script to combine CSVs
    echo -e "${YELLOW}Combining CSV files...${NC}"
    python3 "$PYTHON_SCRIPT" "$LOCAL_FOLDER" \
        "$COMBINED_CSV_NAME"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ CSV combination completed successfully${NC}"
        echo -e "${GREEN}✓ Process completed! Combined CSV saved as: \
            $LOCAL_FOLDER/$COMBINED_CSV_NAME${NC}"
        echo -e "${GREEN}✓ Created at: $(date)${NC}"
    else
        echo -e "${RED}✗ CSV combination failed${NC}"
        exit 1
    fi
    
else
    echo -e "${RED}✗ Rsync failed${NC}"
    exit 1
fi