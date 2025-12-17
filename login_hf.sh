#!/bin/bash

##############################################################################
# Hugging Face Login Script
# 
# This script prompts the user to enter their Hugging Face token and logs in
# to the Hugging Face Hub. The token is securely stored locally.
#
# Usage:
#   ./login_hf.sh                    # Interactive prompt
#   ./login_hf.sh <your-token>       # Pass token as argument
#   HF_TOKEN=<token> ./login_hf.sh   # Pass token as environment variable
##############################################################################

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Hugging Face Hub Authentication${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if token was provided as argument
if [ -n "$1" ]; then
    HF_TOKEN="$1"
    echo -e "${YELLOW}Token provided as argument${NC}"
elif [ -n "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Token provided via HF_TOKEN environment variable${NC}"
else
    # Prompt user for token interactively
    echo -e "${BLUE}Please enter your Hugging Face token:${NC}"
    echo -e "${YELLOW}You can get it at: https://huggingface.co/settings/tokens${NC}"
    echo ""
    
    read -sp "Enter your HF token: " HF_TOKEN
    echo ""
    
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}Error: No token provided${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}Logging in to Hugging Face Hub...${NC}"
echo ""

# Login to Hugging Face using the CLI
if hf auth login --token "$HF_TOKEN" --add-to-git-credential; then
    echo ""
    echo -e "${GREEN}✓ Successfully logged in to Hugging Face Hub!${NC}"
    echo ""
    echo -e "${GREEN}Token information:${NC}"
    hf auth whoami
    echo ""
    echo -e "${BLUE}Token saved at:${NC} ${HOME}/.cache/huggingface/token"
    echo ""
    echo -e "${YELLOW}You can now download SAM3 models and other gated models.${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Login failed. Please check your token and try again.${NC}"
    echo -e "${YELLOW}Make sure your token has read/write permissions.${NC}"
    exit 1
fi
