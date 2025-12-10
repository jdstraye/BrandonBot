# Script to let replit run without interruptions

## systemd to prevent sleep while a replit session is running. This will go away when firefox is closed.
systemd-inhibit --what=idle:sleep --who="Firefox Replit Session" --why="Keeping Replit active overnight" firefox

## Disable Screen Locking
gsettings set org.gnome.desktop.session idle-delay 0

## Prevent Screen Blanking
xset s off         # Disable screen saver
xset -dpms         # Disable DPMS (Energy Star) features
xset s noblank     # Prevent screen blanking

