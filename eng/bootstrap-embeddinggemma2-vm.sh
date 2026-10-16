#!/usr/bin/env bash
# Prepare the dedicated Ubuntu benchmark guest; do not start timed workloads.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
work="$HOME/.cache/onnx-bootstrap"
mkdir -p "$work"
if ! dpkg-query -W packages-microsoft-prod >/dev/null 2>&1; then
    curl --fail --location --retry 3 https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb --output "$work/packages-microsoft-prod.deb"
    sudo -n dpkg -i "$work/packages-microsoft-prod.deb"
fi
sudo -n apt-get update -qq
sudo -n apt-get install -y -qq python3-venv powershell
if ! "$HOME/.dotnet/dotnet" --list-sdks 2>/dev/null | grep -q '^10.0.204 '; then
    curl --fail --location --retry 3 https://dot.net/v1/dotnet-install.sh --output "$work/dotnet-install.sh"
    bash "$work/dotnet-install.sh" --version 10.0.204 --install-dir "$HOME/.dotnet"
fi
sudo -n ln -sfn "$HOME/.dotnet/dotnet" /usr/local/bin/dotnet
dotnet --info
pwsh -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'
python3 --version
printf 'PASS benchmark VM bootstrap\n'
