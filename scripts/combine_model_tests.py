#!/usr/bin/env python3
"""
Combine all CSV files in a directory into a single CSV file.
"""
import pandas as pd
import glob
import os
import sys
import argparse

def combine_test_csvs(folder_path, output_name, add_source_column=True):
    """
    Combine all CSV files in a folder into a single CSV.
    
    Args:
        folder_path: Directory containing CSV files
        output_name: Name of output CSV file
        add_source_column: Whether to add a 'source_file' column
    
    Returns:
        bool: True if successful, False otherwise
    """
    csv_pattern = os.path.join(folder_path, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    # Filter out existing combined_metrics files
    csv_files = [f for f in csv_files 
                 if not os.path.basename(f).startswith('combined_metrics')]
    
    if not csv_files:
        print(f'No CSV files found in {folder_path}'\
              '(excluding combined_metrics files)')
        return False
    
    print(f'Found {len(csv_files)} CSV files ' \
          '(excluding combined_metrics files):')
    for f in csv_files:
        print(f'  - {os.path.basename(f)}')
    
    # Read and combine all CSVs
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add source filename as a column if requested
            if add_source_column:
                df['source_file'] = os.path.basename(file)
            dataframes.append(df)
            print(f'  ✓ Read {len(df)} rows from {os.path.basename(file)}')
        except Exception as e:
            print(f'  ✗ Error reading {file}: {e}')
    
    if not dataframes:
        print('No valid CSV files could be read')
        return False
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save combined CSV
    output_path = os.path.join(folder_path, output_name)
    combined_df.to_csv(output_path, index=False)
    
    print(f'✓ Combined {len(combined_df)} total rows into {output_name}')
    print(f'✓ Saved to: {output_path}')
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine CSV files in a directory'
    )
    parser.add_argument('folder_path', help='Directory containing CSV files')
    parser.add_argument('output_name', help='Name of output CSV file')
    parser.add_argument('--no-source-column', action='store_true', 
                        help='Do not add source_file column')
    
    args = parser.parse_args()
    
    success = combine_test_csvs(
        args.folder_path, 
        args.output_name, 
        add_source_column=not args.no_source_column
    )
    sys.exit(0 if success else 1)