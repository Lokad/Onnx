#!/bin/bash
# Quiet-box bootstrap: Ubuntu 24.04 -> .NET SDK 10.0.204 exact + PowerShell 7 + pip
set -u
LOG=$HOME/bootstrap.log; echo "=== bootstrap start $(date -u +%FT%TZ) ===" | tee $LOG
wget -q https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb >>$LOG 2>&1 && echo MSREPO-OK | tee -a $LOG || echo MSREPO-FAIL | tee -a $LOG
sudo dpkg -i /tmp/packages-microsoft-prod.deb >>$LOG 2>&1
sudo apt-get update -q >>$LOG 2>&1 && echo APT-UPDATE-OK | tee -a $LOG || echo APT-UPDATE-FAIL | tee -a $LOG
sudo apt-get install -y -q powershell python3-pip >>$LOG 2>&1 && echo APT-INSTALL-OK | tee -a $LOG || echo APT-INSTALL-FAIL | tee -a $LOG
wget -q https://dot.net/v1/dotnet-install.sh -O /tmp/dotnet-install.sh >>$LOG 2>&1
chmod +x /tmp/dotnet-install.sh
/tmp/dotnet-install.sh --version 10.0.204 --install-dir $HOME/.dotnet >>$LOG 2>&1 && echo DOTNET-OK | tee -a $LOG || echo DOTNET-FAIL | tee -a $LOG
sudo ln -sf $HOME/.dotnet/dotnet /usr/local/bin/dotnet
grep -q DOTNET_ROOT $HOME/.bashrc 2>/dev/null || echo "export DOTNET_ROOT=\$HOME/.dotnet; export PATH=\$HOME/.dotnet:\$PATH" >> $HOME/.bashrc
echo "=== versions ===" | tee -a $LOG
$HOME/.dotnet/dotnet --list-sdks 2>&1 | tee -a $LOG
pwsh -NoProfile -Command "`$PSVersionTable.PSVersion" 2>&1 | tee -a $LOG
python3 --version 2>&1 | tee -a $LOG; python3 -m pip --version 2>&1 | tee -a $LOG
echo "BOOTSTRAP-DONE $(date -u +%FT%TZ)" | tee -a $LOG
touch $HOME/bootstrap.done
