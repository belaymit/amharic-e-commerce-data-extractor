#!/usr/bin/env python3
"""
Setup script for the restructured Amharic E-commerce Data Extractor project
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is adequate"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_directory_structure():
    """Verify the project structure is correct"""
    required_dirs = [
        'src', 'notebooks', 'scripts', 'data', 'config', 
        'tests', 'docs', 'examples', 'models', 'logs'
    ]
    
    print("Checking directory structure...")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ missing")
            return False
    return True

def check_data_files():
    """Check if essential data files exist"""
    data_files = [
        'data/conll_labeled/amharic_ecommerce_conll.txt',
        'data/conll_labeled/labeling_statistics.json'
    ]
    
    print("\nChecking essential data files...")
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            return False
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\nTesting package imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('seqeval', 'Seqeval')
    ]
    
    failed_imports = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0

def run_demo_test():
    """Run a quick demo test"""
    print("\nRunning demo test...")
    try:
        # Test the demo script
        result = subprocess.run([
            sys.executable, 'scripts/demo_task3_training.py'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Demo script executed successfully")
            return True
        else:
            print(f"❌ Demo script failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Demo script timed out")
        return False
    except Exception as e:
        print(f"❌ Demo script error: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 AMHARIC E-COMMERCE DATA EXTRACTOR")
    print("Project Setup and Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure),
        ("Data Files", check_data_files),
        ("Package Imports", test_imports),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 40)
        if not check_func():
            all_passed = False
            print(f"❌ {check_name} check failed")
    
    if all_passed:
        print("\n🎉 All basic checks passed!")
        
        # Optional demo test
        print("\n🧪 Running optional demo test...")
        if run_demo_test():
            print("\n✅ Setup completed successfully!")
            print("\n📚 Next steps:")
            print("1. Open Jupyter: jupyter notebook notebooks/")
            print("2. Run Task 3: python scripts/demo_task3_training.py")
            print("3. Explore notebooks for Tasks 4 and 5")
        else:
            print("\n⚠️  Demo test failed, but basic setup is complete")
    else:
        print("\n❌ Setup incomplete. Please resolve the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 