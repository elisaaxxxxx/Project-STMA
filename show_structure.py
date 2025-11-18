#!/usr/bin/env python3
"""
📁 Affichage de la Structure du Projet
=====================================

Script utilitaire pour visualiser la structure complète du projet
de stratégie de trading basé sur les moyennes mobiles.
"""

import os
from pathlib import Path
import sys

def get_size_str(size_bytes):
    """Convertit la taille en bytes vers un format lisible"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    return f"{s} {size_names[i]}"

def count_lines(filepath):
    """Compte le nombre de lignes dans un fichier"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except:
        return 0

def show_structure(root_path=None, show_details=False):
    """Affiche la structure du projet"""
    if root_path is None:
        root_path = Path(__file__).parent
    
    root_path = Path(root_path)
    
    print("🚀 " + "="*80)
    print("📊 STRUCTURE DU PROJET DE TRADING - MOVING AVERAGES")
    print("🚀 " + "="*80)
    print(f"📂 Racine: {root_path.absolute()}\n")
    
    # Définir les icônes par type de fichier/dossier
    icons = {
        # Dossiers
        'src': '🔧',
        'data': '📊',
        'raw': '📥',
        'processed': '⚙️',
        'results': '📈',
        'backtest': '🎯',
        'variations': '🔬',
        '__pycache__': '🗑️',
        '.venv': '🐍',
        
        # Extensions de fichiers
        '.py': '🐍',
        '.csv': '📄',
        '.png': '🖼️',
        '.md': '📖',
        '.txt': '📝',
        '.json': '⚙️',
        '.yml': '⚙️',
        '.yaml': '⚙️',
    }
    
    def get_icon(path):
        if path.is_dir():
            return icons.get(path.name, '📁')
        else:
            return icons.get(path.suffix, '📄')
    
    def print_tree(path, prefix="", is_last=True):
        if path.name.startswith('.') and path.name not in ['.venv']:
            return  # Skip hidden files except .venv
        
        # Construire le préfixe pour l'affichage
        connector = "└── " if is_last else "├── "
        icon = get_icon(path)
        
        # Nom du fichier/dossier
        display_name = f"{icon} {path.name}"
        
        # Ajouter des informations supplémentaires si demandé
        if show_details and path.is_file():
            size = get_size_str(path.stat().st_size)
            if path.suffix == '.py':
                lines = count_lines(path)
                display_name += f" ({lines} lignes, {size})"
            elif path.suffix == '.csv':
                display_name += f" ({size})"
            else:
                display_name += f" ({size})"
        
        print(f"{prefix}{connector}{display_name}")
        
        # Si c'est un dossier, afficher son contenu
        if path.is_dir() and path.name != '__pycache__':
            try:
                children = sorted([p for p in path.iterdir() 
                                 if not p.name.startswith('.') or p.name == '.venv'])
                for i, child in enumerate(children):
                    is_last_child = i == len(children) - 1
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    print_tree(child, new_prefix, is_last_child)
            except PermissionError:
                pass
    
    # Afficher l'arborescence
    print_tree(root_path)
    
    # Statistiques
    print("\n" + "="*80)
    print("📊 STATISTIQUES DU PROJET")
    print("="*80)
    
    stats = {
        'python_files': 0,
        'csv_files': 0,
        'total_files': 0,
        'total_dirs': 0,
        'python_lines': 0
    }
    
    for path in root_path.rglob('*'):
        if path.is_file():
            stats['total_files'] += 1
            if path.suffix == '.py':
                stats['python_files'] += 1
                stats['python_lines'] += count_lines(path)
            elif path.suffix == '.csv':
                stats['csv_files'] += 1
        elif path.is_dir():
            stats['total_dirs'] += 1
    
    print(f"🐍 Fichiers Python      : {stats['python_files']} ({stats['python_lines']} lignes)")
    print(f"📄 Fichiers CSV         : {stats['csv_files']}")
    print(f"📁 Dossiers            : {stats['total_dirs']}")
    print(f"📋 Total fichiers      : {stats['total_files']}")
    
    # Configuration actuelle
    try:
        sys.path.append(str(root_path))
        from project_config import TICKERS, START_DATE, END_DATE
        print(f"\n⚙️  Configuration actuelle:")
        print(f"   📊 Tickers: {', '.join(TICKERS)}")
        print(f"   📅 Période: {START_DATE} → {END_DATE}")
    except ImportError:
        print("\n⚠️  Impossible de lire la configuration")
    
    print("\n" + "="*80)
    print("✅ Utilisez 'python run_pipeline.py --all' pour lancer l'analyse complète")
    print("📖 Consultez README.md pour plus de détails")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Affiche la structure du projet")
    parser.add_argument("--details", "-d", action="store_true", 
                       help="Afficher les détails (taille, lignes)")
    parser.add_argument("--path", "-p", type=str, default=None,
                       help="Chemin racine (par défaut: dossier courant)")
    
    args = parser.parse_args()
    show_structure(args.path, args.details)