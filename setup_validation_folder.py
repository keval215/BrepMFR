#!/usr/bin/env python3
"""
Setup validation folder from larger dataset
Copies files listed in val.txt from source directory to val/ folder
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def setup_validation_folder(
    source_dir,           # Directory containing all 100k files
    val_txt_path="data/val.txt",
    val_folder="data/val",
    file_extensions=[".bin", ".json", ".stp"]
):
    """
    Create validation folder with files matching val.txt
    
    Args:
        source_dir: Path to directory containing all original files
        val_txt_path: Path to val.txt file
        val_folder: Path to validation folder to create
        file_extensions: File extensions to copy
    """
    
    source_path = Path(source_dir)
    val_path = Path(val_folder)
    
    # Create validation folder
    val_path.mkdir(exist_ok=True)
    print(f"Created validation folder: {val_path}")
    
    # Read validation file IDs
    with open(val_txt_path, "r") as f:
        val_file_ids = [x.strip() for x in f.readlines()]
    
    print(f"Found {len(val_file_ids)} file IDs in {val_txt_path}")
    print(f"First few IDs: {val_file_ids[:5]}")
    print(f"Last few IDs: {val_file_ids[-5:]}")
    
    # Copy files for each validation ID
    copied_files = 0
    missing_files = []
    
    for file_id in tqdm(val_file_ids, desc="Copying validation files"):
        for ext in file_extensions:
            source_file = source_path / f"{file_id}{ext}"
            dest_file = val_path / f"{file_id}{ext}"
            
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
                copied_files += 1
            else:
                missing_files.append(str(source_file))
    
    print(f"\n=== Results ===")
    print(f"Files copied: {copied_files}")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\nMissing files (first 10):")
        for missing in missing_files[:10]:
            print(f"  - {missing}")
    
    # Verify validation folder
    val_bin_files = list(val_path.glob("*.bin"))
    val_json_files = list(val_path.glob("*.json"))
    
    print(f"\nValidation folder contents:")
    print(f"  .bin files: {len(val_bin_files)}")
    print(f"  .json files: {len(val_json_files)}")
    print(f"  Expected: {len(val_file_ids)} of each type")
    
    if len(val_bin_files) == len(val_file_ids):
        print("✅ Validation folder setup complete!")
    else:
        print("⚠️  Some files are missing. Check the source directory path.")
    
    return len(val_bin_files), len(missing_files)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser("Setup BrepMFR validation folder")
    parser.add_argument("--source_dir", required=True, help="Directory containing all original files")
    parser.add_argument("--val_txt", default="data/val.txt", help="Path to val.txt file")
    parser.add_argument("--val_folder", default="data/val", help="Validation folder to create")
    
    args = parser.parse_args()
    
    print("=== BrepMFR Validation Folder Setup ===")
    print(f"Source directory: {args.source_dir}")
    print(f"Val.txt file: {args.val_txt}")
    print(f"Validation folder: {args.val_folder}")
    
    # Check if source directory exists
    if not Path(args.source_dir).exists():
        print(f"❌ Source directory does not exist: {args.source_dir}")
        return
    
    # Check if val.txt exists
    if not Path(args.val_txt).exists():
        print(f"❌ val.txt file does not exist: {args.val_txt}")
        return
    
    # Setup validation folder
    copied, missing = setup_validation_folder(
        source_dir=args.source_dir,
        val_txt_path=args.val_txt,
        val_folder=args.val_folder
    )
    
    if missing == 0:
        print("\n🎉 Validation folder setup successful!")
        print("You can now run training with this validation set.")
    else:
        print(f"\n⚠️  Setup completed with {missing} missing files.")
        print("Check if the source directory contains all required files.")

if __name__ == "__main__":
    main()