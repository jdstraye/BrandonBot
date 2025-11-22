#!/bin/bash

# Ensure ~/.ssh directory exists
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Symlink keys from persistent storage
if [ -f .ssh-keys/replit ]; then
  ln -sf $(pwd)/.ssh-keys/replit ~/.ssh/replit
  ln -sf $(pwd)/.ssh-keys/replit.pub ~/.ssh/replit.pub
fi

if [ -f .ssh-keys/github ]; then
  ln -sf $(pwd)/.ssh-keys/github ~/.ssh/github
  ln -sf $(pwd)/.ssh-keys/github.pub ~/.ssh/github.pub
fi

# Setup SSH config
cat > ~/.ssh/config << 'EOF'
Host *.replit.dev
    Port 22
    IdentityFile ~/.ssh/replit
    StrictHostKeyChecking accept-new

Host github.com
    IdentityFile ~/.ssh/github
    StrictHostKeyChecking accept-new
EOF

chmod 600 ~/.ssh/config

echo "SSH keys restored from persistent storage"