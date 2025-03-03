import click
import pandas as pd
import os

def get_external_type(x):
    if pd.isna(x):
        return None
    code = str(x)[0].lower()
    if code == 'b':
        return "bank"
    elif code == 'm':
        return "merchant"
    elif code == 'c':
        return "customer"
    else:
        return "other"

@click.command()
@click.option('--input_path', required=True, help='Path to the input CSV file')
@click.option('--output_path', required=True, help='Path to write the processed CSV file')
def preprocess(input_path, output_path):
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Check if the "External" column is present
    if "External" in df.columns:
        df["External_Type"] = df["External"].apply(get_external_type)
        print("Added External_Type column.")
    else:
        print("The 'External' column was not found in the input file.")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, compression='gzip')
    print(f"Processed file written to {output_path}")

if __name__ == "__main__":
    preprocess()