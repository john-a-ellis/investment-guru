# fix_model_names.py
# This script helps fix model filenames by ensuring the symbol is included in the filename

import os
import sys
import pickle
import shutil
from datetime import datetime

def scan_model_directory():
    """Scan the models directory and return information about all model files"""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("No model files found in the models directory")
        return []
    
    model_info = []
    for model_file in model_files:
        full_path = os.path.join(models_dir, model_file)
        file_size = os.path.getsize(full_path) / 1024  # Size in KB
        file_date = datetime.fromtimestamp(os.path.getmtime(full_path))
        
        # Check if filename includes a symbol
        parts = model_file.split('_')
        if len(parts) > 1:
            model_type = parts[0]
            symbol = parts[1].split('.pkl')[0]
        else:
            model_type = parts[0].split('.pkl')[0]
            symbol = None
        
        model_info.append({
            'filename': model_file,
            'path': full_path,
            'size_kb': file_size,
            'modified': file_date,
            'model_type': model_type,
            'symbol': symbol
        })
    
    return model_info

def check_for_symbol_issues(model_info, symbols):
    """Check for models that are missing symbols in their filenames"""
    issues = []
    
    # Find models without symbols in filename
    for model in model_info:
        if not model['symbol'] or model['symbol'] == 'None':
            issues.append({
                'model': model,
                'issue': 'missing_symbol',
                'suggested_symbols': symbols
            })
    
    return issues

def fix_model_filename(model_path, new_symbol):
    """Fix a model filename by adding the correct symbol"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    # Get directory and filename
    model_dir = os.path.dirname(model_path)
    old_filename = os.path.basename(model_path)
    
    # Generate new filename with symbol
    model_type = old_filename.split('_')[0] if '_' in old_filename else old_filename.split('.')[0]
    new_filename = f"{model_type}_{new_symbol}.pkl"
    new_path = os.path.join(model_dir, new_filename)
    
    # Check if new file already exists
    if os.path.exists(new_path):
        print(f"Warning: {new_filename} already exists, will create backup")
        backup_path = f"{new_path}.backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(new_path, backup_path)
    
    # Copy the file to new name
    print(f"Copying {old_filename} to {new_filename}")
    shutil.copy2(model_path, new_path)
    
    # Verify the new file
    if os.path.exists(new_path):
        print(f"Successfully created {new_filename}")
        
        # Option to delete old file
        if input(f"Delete original file {old_filename}? (y/n): ").lower() == 'y':
            os.remove(model_path)
            print(f"Deleted {old_filename}")
        
        return True
    else:
        print(f"Failed to create {new_filename}")
        return False

def update_model_object(model_path, symbol):
    """Try to update the symbol attribute inside the model object itself"""
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Check if model has a symbol attribute
        if hasattr(model, 'symbol'):
            print(f"Current symbol in model: {model.symbol}")
            model.symbol = symbol
            print(f"Updated model symbol to: {symbol}")
            
            # Save the updated model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Saved updated model with new symbol")
            return True
    except Exception as e:
        print(f"Error updating model object: {e}")
    
    return False

def verify_model_files(fixed_models):
    """Verify that fixed model files can be loaded properly"""
    for model_path in fixed_models:
        try:
            print(f"Verifying {model_path}...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Check model type and symbol
            if hasattr(model, 'symbol'):
                print(f"  Model symbol: {model.symbol}")
            
            print(f"  Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"  Error loading model: {e}")

def main():
    # Get symbols from the user
    print("Enter symbols that should be associated with your models (separated by commas):")
    symbols_input = input("> ")
    symbols = [s.strip() for s in symbols_input.split(',')]
    
    # Scan model directory
    print("\nScanning models directory...")
    model_info = scan_model_directory()
    
    if not model_info:
        print("No models found. Exiting.")
        return
    
    print(f"\nFound {len(model_info)} model files:")
    for i, model in enumerate(model_info, 1):
        symbol_info = f" - Symbol: {model['symbol']}" if model['symbol'] else " - No symbol in filename"
        print(f"{i}. {model['filename']} ({model['size_kb']:.1f} KB){symbol_info}")
    
    # Check for issues
    issues = check_for_symbol_issues(model_info, symbols)
    
    if not issues:
        print("\nNo symbol issues found in model filenames.")
        return
    
    print(f"\nFound {len(issues)} models with symbol issues:")
    for i, issue in enumerate(issues, 1):
        model = issue['model']
        print(f"{i}. {model['filename']} - {issue['issue']}")
    
    # Fix issues
    print("\nWould you like to fix these issues? (y/n)")
    if input("> ").lower() != 'y':
        return
    
    fixed_models = []
    
    for issue in issues:
        model = issue['model']
        print(f"\nFixing {model['filename']}...")
        
        if len(issue['suggested_symbols']) == 1:
            symbol = issue['suggested_symbols'][0]
            print(f"Using symbol: {symbol}")
        else:
            print("Select a symbol for this model:")
            for i, symbol in enumerate(issue['suggested_symbols'], 1):
                print(f"{i}. {symbol}")
            print(f"{len(issue['suggested_symbols'])+1}. Enter a different symbol")
            
            choice = input("> ")
            if choice.isdigit() and 1 <= int(choice) <= len(issue['suggested_symbols']):
                symbol = issue['suggested_symbols'][int(choice)-1]
            else:
                symbol = input("Enter symbol: ")
        
        # Fix the filename
        new_path = os.path.join("models", f"{model['model_type']}_{symbol}.pkl")
        if fix_model_filename(model['path'], symbol):
            fixed_models.append(new_path)
            # Try to update the model object itself
            update_model_object(new_path, symbol)
    
    # Verify fixed models
    if fixed_models:
        print("\nVerifying fixed models...")
        verify_model_files(fixed_models)
        
        print("\nAll models have been fixed and verified.")
    else:
        print("\nNo models were fixed.")

if __name__ == "__main__":
    print("===== MODEL FILENAME FIXER =====\n")
    main()