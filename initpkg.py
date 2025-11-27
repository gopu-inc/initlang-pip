#!/usr/bin/env python3
"""
Gestionnaire de paquets INIT - Pour INITLANG
"""

import os
import sys
import json
import shutil
import argparse
import urllib.request
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# ==================== CONFIGURATION ====================

INITLANG_HOME = Path.home() / ".initlang"
PACKAGES_DIR = INITLANG_HOME / "packages"
CACHE_DIR = INITLANG_HOME / "cache"
CONFIG_FILE = INITLANG_HOME / "config.json"

# Repository par défaut
DEFAULT_REPOSITORY = "https://raw.githubusercontent.com/gopu-inc/initlang-packages/main"

# ==================== CLASSES PRINCIPALES ====================

class PackageManager:
    def __init__(self):
        self.setup_directories()
        self.load_config()
    
    def setup_directories(self):
        """Crée les répertoires nécessaires"""
        for directory in [INITLANG_HOME, PACKAGES_DIR, CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """Charge la configuration"""
        self.config = {
            "repository": DEFAULT_REPOSITORY,
            "installed_packages": {}
        }
        
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
    
    def fetch_package_index(self) -> Dict:
        """Récupère l'index des paquets"""
        cache_file = CACHE_DIR / "index.json"
        
        # Utiliser le cache si récent (5 minutes)
        if cache_file.exists():
            import time
            if time.time() - cache_file.stat().st_mtime < 300:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        try:
            url = f"{self.config['repository']}/index.json"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            # Sauvegarder dans le cache
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return data
        except Exception as e:
            print(f"Error fetching package index: {e}")
            return {}
    
    def install(self, package_names: List[str]):
        """Installe un ou plusieurs paquets"""
        index = self.fetch_package_index()
        
        for package_name in package_names:
            self._install_single(package_name, index)
    
    def _install_single(self, package_name: str, index: Dict):
        """Installe un paquet unique"""
        if package_name in self.config["installed_packages"]:
            print(f"Package '{package_name}' is already installed")
            return
        
        if package_name not in index:
            print(f"Package '{package_name}' not found in repository")
            return
        
        package_info = index[package_name]
        print(f"Installing {package_name} v{package_info['version']}...")
        
        try:
            # Télécharger le paquet
            package_url = f"{self.config['repository']}/packages/{package_name}.init"
            with urllib.request.urlopen(package_url) as response:
                package_content = response.read().decode()
            
            # Créer le répertoire du paquet
            package_dir = PACKAGES_DIR / package_name
            package_dir.mkdir(exist_ok=True)
            
            # Sauvegarder le fichier principal
            main_file = package_dir / "main.init"
            with open(main_file, 'w') as f:
                f.write(package_content)
            
            # Sauvegarder les métadonnées
            meta_file = package_dir / "package.json"
            with open(meta_file, 'w') as f:
                json.dump(package_info, f, indent=2)
            
            # Mettre à jour la configuration
            self.config["installed_packages"][package_name] = {
                "version": package_info["version"],
                "path": str(package_dir)
            }
            
            self.save_config()
            print(f"✓ {package_name} v{package_info['version']} installed successfully")
            
        except Exception as e:
            print(f"✗ Failed to install {package_name}: {e}")
    
    def uninstall(self, package_names: List[str]):
        """Désinstalle un ou plusieurs paquets"""
        for package_name in package_names:
            self._uninstall_single(package_name)
    
    def _uninstall_single(self, package_name: str):
        """Désinstalle un paquet unique"""
        if package_name not in self.config["installed_packages"]:
            print(f"Package '{package_name}' is not installed")
            return
        
        try:
            # Supprimer le répertoire du paquet
            package_dir = PACKAGES_DIR / package_name
            if package_dir.exists():
                shutil.rmtree(package_dir)
            
            # Mettre à jour la configuration
            del self.config["installed_packages"][package_name]
            self.save_config()
            
            print(f"✓ {package_name} uninstalled successfully")
            
        except Exception as e:
            print(f"✗ Failed to uninstall {package_name}: {e}")
    
    def list_installed(self):
        """Liste les paquets installés"""
        if not self.config["installed_packages"]:
            print("No packages installed")
            return
        
        print("Installed packages:")
        for name, info in self.config["installed_packages"].items():
            print(f"  {name} v{info['version']}")
    
    def list_available(self):
        """Liste les paquets disponibles"""
        index = self.fetch_package_index()
        
        if not index:
            print("No packages available in repository")
            return
        
        print("Available packages:")
        for name, info in index.items():
            installed = "(installed)" if name in self.config["installed_packages"] else ""
            print(f"  {name} v{info['version']} {installed}")
    
    def search(self, query: str):
        """Recherche des paquets"""
        index = self.fetch_package_index()
        
        results = []
        for name, info in index.items():
            if query.lower() in name.lower() or query.lower() in info.get("description", "").lower():
                results.append((name, info))
        
        if not results:
            print(f"No packages found for '{query}'")
            return
        
        print(f"Search results for '{query}':")
        for name, info in results:
            installed = "(installed)" if name in self.config["installed_packages"] else ""
            print(f"  {name} v{info['version']} - {info.get('description', '')} {installed}")
    
    def update(self):
        """Met à jour l'index des paquets"""
        cache_file = CACHE_DIR / "index.json"
        if cache_file.exists():
            cache_file.unlink()
        
        index = self.fetch_package_index()
        print(f"Package index updated ({len(index)} packages available)")
    
    def info(self, package_name: str):
        """Affiche les informations d'un paquet"""
        index = self.fetch_package_index()
        
        if package_name in index:
            info = index[package_name]
            print(f"Package: {package_name}")
            print(f"Version: {info['version']}")
            print(f"Description: {info.get('description', 'No description')}")
            print(f"Author: {info.get('author', 'Unknown')}")
            
            if package_name in self.config["installed_packages"]:
                print("Status: Installed")
            else:
                print("Status: Not installed")
        else:
            print(f"Package '{package_name}' not found")

# ==================== CLI ====================

def show_help():
    print("""
INIT Package Manager

Usage:
  initpkg [COMMAND] [ARGS]

Commands:
  install <package...>    Install packages
  uninstall <package...>  Uninstall packages
  list                    List installed packages
  search <query>          Search packages
  update                  Update package index
  info <package>          Show package information
  available               List available packages

Examples:
  initpkg install math strings
  initpkg search http
  initpkg list
  initpkg update
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
        if args.command == 'install':
            if not args.args:
                print("Error: Please specify packages to install")
                return
            pm.install(args.args)
        
        elif args.command == 'uninstall':
            if not args.args:
                print("Error: Please specify packages to uninstall")
                return
            pm.uninstall(args.args)
        
        elif args.command == 'list':
            pm.list_installed()
        
        elif args.command == 'available':
            pm.list_available()
        
        elif args.command == 'search':
            if not args.args:
                print("Error: Please specify search query")
                return
            pm.search(args.args[0])
        
        elif args.command == 'update':
            pm.update()
        
        elif args.command == 'info':
            if not args.args:
                print("Error: Please specify package name")
                return
            pm.info(args.args[0])
        
        else:
            print(f"Unknown command: {args.command}")
            show_help()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
