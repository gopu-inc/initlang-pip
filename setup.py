#!/usr/bin/env python3
"""
Setup script for INITLANG - Innovative Programming Language
"""

from setuptools import setup, find_packages
import os
import re

# Read the version from the main file
def get_version():
    try:
        with open('initlang.py', 'r', encoding='utf-8') as f:
            content = f.read()
            version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", content)
            if version_match:
                return version_match.group(1)
    except Exception:
        pass
    return "1.0.0"

def get_long_description():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return "INITLANG - Innovative programming language with modern syntax"

def get_requirements():
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception:
        return []

# Configuration du package
setup(
    name="initlang",
    version="1.0.0",
    
    # Métadonnées
    author="Mauricio-100",
    author_email="ceoseshell@gmail.com",
    description="INITLANG - Innovative programming language with modern syntax",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords="programming, language, compiler, interpreter, initlang",
    url="https://github.com/gopu-inc/initlang-pip",
    
    # Classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Interpreters",
        "Topic :: Software Development :: Compilers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Packages
    py_modules=["initlang"],
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    
    # Dépendances
    python_requires=">=3.7",
    install_requires=get_requirements(),
    
    # Scripts d'entrée
    entry_points={
        "console_scripts": [
            "initlang=deps:main", # Alias court
        ],
    },
    
    # Données supplémentaires
    include_package_data=True,
    package_data={
        "initlang": [
            "examples/*.init",
            "stdlib/*.init",
        ],
    },
    
    # Métadonnées supplémentaires
    license="MIT",
    project_urls={
        "Documentation": "https://github.com/gopu-inc/initlang-pip/docs",
        "Source": "https://github.com/gopu-inc/initlang-pip",
        "Tracker": "https://github.com/gopu-inc/initlang-pip/issues",
    },
    
    # Options de build
    zip_safe=False,
)
