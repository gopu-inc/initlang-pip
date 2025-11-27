#!/usr/bin/env python3
"""
Gestionnaire de paquets INIT - Pour INITLANG
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List

# Configuration
INITLANG_HOME = Path.home() / ".initlang"
PACKAGES_DIR = INITLANG_HOME / "packages"
CONFIG_FILE = INITLANG_HOME / "packages.json"

class PackageManager:
    def __init__(self):
        self.setup_directories()
        self.load_config()
    
    def setup_directories(self):
        """Crée les répertoires nécessaires"""
        PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """Charge la configuration"""
        self.config = {"installed_packages": {}}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    self.config.update(json.load(f))
            except:
                pass
    
    def save_config(self):
        """Sauvegarde la configuration"""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def create_package(self, package_name: str, version: str = "1.0.0"):
        """Crée un nouveau paquet local"""
        package_dir = PACKAGES_DIR / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Fichier principal
        main_file = package_dir / "main.init"
        if not main_file.exists():
            with open(main_file, 'w') as f:
                f.write(f"""# Package {package_name}

init.log("Package {package_name} loaded!")

fi hello() {{
    init.ger("Hello from {package_name}!")
}}

let version ==> "{version}"
""")
        
        # Métadonnées
        meta_file = package_dir / "package.json"
        with open(meta_file, 'w') as f:
            json.dump({
                "name": package_name,
                "version": version,
                "description": f"Package {package_name} for INITLANG"
            }, f, indent=2)
        
        print(f"✓ Package '{package_name}' created at {package_dir}")
    
    def install_local(self, package_path: str):
        """Installe un paquet local"""
        source_dir = Path(package_path)
        package_name = source_dir.name
        
        if not (source_dir / "main.init").exists():
            print(f"✗ No main.init found in {package_path}")
            return
        
        # Copier le paquet
        target_dir = PACKAGES_DIR / package_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        shutil.copytree(source_dir, target_dir)
        
        # Mettre à jour la configuration
        self.config["installed_packages"][package_name] = {
            "version": "1.0.0",
            "path": str(target_dir)
        }
        self.save_config()
        
        print(f"✓ Package '{package_name}' installed")
    
    def list_installed(self):
        """Liste les paquets installés"""
        if not self.config["installed_packages"]:
            print("No packages installed")
            return
        
        print("Installed packages:")
        for name, info in self.config["installed_packages"].items():
            print(f"  {name} v{info['version']}")
    
    def uninstall(self, package_name: str):
        """Désinstalle un paquet"""
        if package_name not in self.config["installed_packages"]:
            print(f"Package '{package_name}' is not installed")
            return
        
        package_dir = PACKAGES_DIR / package_name
        if package_dir.exists():
            shutil.rmtree(package_dir)
        
        del self.config["installed_packages"][package_name]
        self.save_config()
        
        print(f"✓ Package '{package_name}' uninstalled")

def show_help():
    print("""
INIT Package Manager

Usage:
  initpkg [COMMAND] [ARGS]

Commands:
  create <name>        Create a new package
  install <path>       Install local package
  list                 List installed packages
  uninstall <name>     Uninstall package

Examples:
  initpkg create math
  initpkg install ./my-package
  initpkg list
  initpkg uninstall math
    """)

def main():
    parser = argparse.ArgumentParser(description="INIT Package Manager", add_help=False)
    parser.add_argument('command', nargs='?', help='Command to execute')
    parser.add_argument('args', nargs='*', help='Command arguments')
    
    args = parser.parse_args()
    
    pm = PackageManager()
    
    if not args.command or args.command in ['-h', '--help']:
        show_help()
        return
    
    try:
        if args.command == 'create':
            if not args.args:
                print("Error: Please specify package name")
                return
            pm.create_package(args.args[0])
        
        elif args.command == 'install':
            if not args.args:
                print("Error: Please specify package path")
                return
            pm.install_local(args.args[0])
        
        elif args.command == 'list':
            pm.list_installed()
        
        elif args.command == 'uninstall':
            if not args.args:
                print("Error: Please specify package name")
                return
            pm.uninstall(args.args[0])
        
        else:
            print(f"Unknown command: {args.command}")
            show_help()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
