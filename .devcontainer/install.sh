#!/bin/bash
set -e  # Arrête le script si une commande échoue

# Installer ScimBa en mode éditable
sudo /home/firedrake/firedrake/bin/python3 -m pip install --no-deps -e /usr/src/app/ScimBa/.

# Installer ton projet en mode éditable
sudo /home/firedrake/firedrake/bin/python3 -m pip install -e /usr/src/app/Article1/code/.

# Lancer bash ou tout autre commande après installation
exec "$@"