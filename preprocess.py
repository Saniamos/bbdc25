import importlib
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
@click.option('--input_path', default="./task_orig/val_set/x_val.csv", required=True, help='Path to the input CSV file')
@click.option('--output_path', default="./task_orig/val_set/y_val.csv", required=True, help='Path to write the processed CSV file')
@click.option('--ft_module', required=False, default=None, help='Python module containing the Model class')
def preprocess(input_path, output_path, ft_module):
    # Dynamically import the module containing the Model class.
    fts = None
    if ft_module is not None:
        try:
            mod = importlib.import_module(ft_module)
            fts = getattr(mod, "Features")
            print(f"Imported Model from module {ft_module}")
        except Exception as e:
            print(f"Error importing Model from module {ft_module}: {e}")
            return
    
    # Read the CSV file
    df = pd.read_csv(input_path)

    if 'AccountID' in df.columns:
        # filter out the non-fraud spammer in val set
        df = df[df['AccountID'] != 'C5686050095']
    
    # Check if the "External" column is present
    if "External" in df.columns:
        df["External_Type"] = df["External"].apply(get_external_type)
        if fts is not None and hasattr(fts, "extract"):
            df = fts().extract(df)
            val_counts = df.groupby('AccountID').size().value_counts()
            print('Account Transaction Sequence Lengths:')
            print(val_counts)
            print('Longest Transaction Sequence:', val_counts.index.max())
        print("Added External_Type column.")
    else:
        print("The 'External' column was not found in the input file.")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Remove the output file if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file at {output_path}")
    
    df.to_parquet(output_path, index=False)
    print(f"Processed file written to {output_path}")

if __name__ == "__main__":
    preprocess()