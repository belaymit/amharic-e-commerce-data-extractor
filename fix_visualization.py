#!/usr/bin/env python3
"""
Fix visualization notebook by:
1. Handling float prices safely (no .replace() on floats)
2. Separating combined graphs into individual displays
"""

import json
import re

def safe_price_conversion_code():
    """Return the safe price conversion function code."""
    return '''def safe_price_conversion(price):
    """Safely convert price to float, handling both strings and numbers."""
    if isinstance(price, (int, float)):
        return float(price)
    try:
        # Remove commas and convert to float
        clean_price = str(price).replace(',', '').strip()
        if clean_price.replace('.', '').isdigit():
            return float(clean_price)
    except:
        pass
    return None'''

def fix_price_processing_in_code(code):
    """Fix price processing code to handle floats safely."""
    # Replace problematic price processing patterns
    patterns_to_fix = [
        r'avg_price = np\.mean\(\[float\(p\.replace\([^)]+\)\) for p in all_prices if p\.replace\([^)]+\)\.isdigit\(\)\]\)',
        r'price_val = float\(price\.replace\([^)]+\)\)',
        r'clean_price = str\(price\)\.replace\([^)]+\)',
    ]
    
    # Replace with safe conversion
    for pattern in patterns_to_fix:
        code = re.sub(pattern, 'converted_price = safe_price_conversion(price)\n                if converted_price is not None:\n                    prices.append(converted_price)', code)
    
    # Fix the specific avg_price calculation
    if 'avg_price = np.mean([float(p.replace' in code:
        code = code.replace(
            "avg_price = np.mean([float(p.replace(',', '')) for p in all_prices if p.replace(',', '').isdigit()])",
            """# Clean prices safely handling both floats and strings
    cleaned_prices = []
    for p in all_prices:
        converted_p = safe_price_conversion(p)
        if converted_p is not None:
            cleaned_prices.append(converted_p)
    
    avg_price = np.mean(cleaned_prices) if cleaned_prices else 0"""
        )
    
    return code

def separate_subplots_in_code(code):
    """Convert subplot code to individual plot displays."""
    # Remove subplot creation
    code = re.sub(r'fig, axes = plt\.subplots\([^)]+\)', '', code)
    code = re.sub(r'fig\.suptitle\([^)]+\)', '', code)
    
    # Replace axes[x, y] with plt.figure() and plt
    subplot_replacements = [
        (r'axes\[(\d+), (\d+)\]\.', 'plt.figure(figsize=(12, 6))\n    plt.'),
        (r'axes\[(\d+)\]\.', 'plt.figure(figsize=(12, 6))\n    plt.'),
    ]
    
    for pattern, replacement in subplot_replacements:
        code = re.sub(pattern, replacement, code)
    
    # Add plt.show() after each plot
    code = re.sub(r'(plt\.(bar|hist|plot|pie|boxplot)\([^}]+\}?\))', r'\1\n    plt.tight_layout()\n    plt.show()', code)
    
    return code

def fix_notebook():
    """Fix the visualization notebook."""
    print("🔧 Fixing visualization notebook...")
    
    # Read the notebook
    with open('notebook/06_data_visualization.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Add safe price conversion function to the beginning
    safe_conversion_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            safe_price_conversion_code() + "\n\nprint(\"✅ Safe price conversion function loaded\")"
        ]
    }
    
    # Insert after imports
    notebook['cells'].insert(2, safe_conversion_cell)
    
    # Fix each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_code = ''.join(cell['source'])
            
            # Fix price processing issues
            if '.replace(' in source_code and ('price' in source_code or 'float(' in source_code):
                source_code = fix_price_processing_in_code(source_code)
            
            # Separate subplots
            if 'axes[' in source_code or 'fig, axes' in source_code:
                source_code = separate_subplots_in_code(source_code)
                # Add individual graph headers
                if 'display_channel_analysis' in source_code:
                    source_code = source_code.replace(
                        'def display_channel_analysis():',
                        'def display_channel_analysis():\n    """Display channel analysis with individual graphs."""\n    \n    print("📊 Channel Analysis Dashboard")\n    print("=" * 40)'
                    )
                elif 'display_price_analysis' in source_code:
                    source_code = source_code.replace(
                        'def display_price_analysis():',
                        'def display_price_analysis():\n    """Display price analysis with individual graphs."""\n    \n    print("💰 Price Analysis Dashboard")\n    print("=" * 40)'
                    )
            
            # Update the cell source
            cell['source'] = source_code.split('\n')
            # Add newlines back
            cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line for i, line in enumerate(cell['source'])]
    
    # Write the fixed notebook
    with open('notebook/06_data_visualization.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Visualization notebook fixed!")
    print("- Added safe price conversion function")
    print("- Fixed float AttributeError issues")
    print("- Separated combined graphs into individual displays")
    print("- Enhanced graph formatting and styling")

if __name__ == "__main__":
    fix_notebook() 