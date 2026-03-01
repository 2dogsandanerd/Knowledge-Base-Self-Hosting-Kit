#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}   Knowledge Base Self-Hosting Kit - Setup     ${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# 1. Check for .env
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from example...${NC}"
    cp .env.example .env
else
    echo -e "${GREEN}Found existing .env file.${NC}"
fi

# 2. Configure Document Path
current_path=$(grep "DOCS_DIR=" .env | cut -d '=' -f2)
echo -e "\n${YELLOW}Where are your documents located?${NC}"
echo -e "Current setting: ${GREEN}${current_path:-./data/docs}${NC}"
echo -e "Enter full path (or press Enter to keep current):"
read -r user_path

if [ ! -z "$user_path" ]; then
    # Escape special characters for sed
    escaped_path=$(printf '%s\n' "$user_path" | sed -e 's/[\/&]/\\&/g')
    
    # Update .env (cross-platform compatible sed)
    if grep -q "DOCS_DIR=" .env; then
        sed -i "s/^DOCS_DIR=.*/DOCS_DIR=$escaped_path/" .env
    else
        echo "DOCS_DIR=$user_path" >> .env
    fi
    echo -e "${GREEN}Updated DOCS_DIR to: $user_path${NC}"
else
    echo -e "${BLUE}Keeping current path.${NC}"
fi

# Ensure the directory exists (to avoid Docker creating it as root)
final_path=$(grep "DOCS_DIR=" .env | cut -d '=' -f2)
if [ -z "$final_path" ]; then
    final_path="./data/docs"
fi

if [ ! -d "$final_path" ]; then
    echo -e "\n${YELLOW}Creating documents directory at: $final_path${NC}"
    mkdir -p "$final_path"
    echo -e "${GREEN}Directory created with your user permissions.${NC}"
else
    echo -e "\n${GREEN}Documents directory exists at: $final_path${NC}"
fi

# 3. Create cache volume directory if it doesn't exist (optional, docker handles it usually)
# But good to ensure permissions if we were mapping to host
# echo -e "\n${YELLOW}Ensuring cache volume exists...${NC}"

# 4. Start/Restart Docker
echo -e "\n${YELLOW}Ready to apply changes. This will restart the container.${NC}"
read -p "Start/Restart Docker now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${BLUE}Stopping containers...${NC}"
    docker compose down
    
    echo -e "\n${BLUE}Building and starting...${NC}"
    docker compose up -d --build
    
    echo -e "\n${GREEN}✅ Done! System is running.${NC}"
    echo -e "UI: http://localhost:8080"
    echo -e "API: http://localhost:8080/docs"
else
    echo -e "\n${BLUE}Skipping restart. Run 'docker compose up -d' manually when ready.${NC}"
fi
