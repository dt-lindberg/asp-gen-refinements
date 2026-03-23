"""Helper script to look for hardcoded solutions in the ASPs, serves as initial attempt at doing this programmatically.

* Load data from mistakes/.xlsx files, look at 'refinement_i' columns (i=[0,...,n])
"""

import os
import pandas

MISTAKES_FILE = "mistakes_0318_103804.xlsx"

if __name__ == "__main__":
    # Load data
    df = pandas.read_excel(
        os.path.join(os.path.dirname(__file__), f"../mistakes/{MISTAKES_FILE}"),
        sheet_name="results",
        header=0,
        engine="openpyxl",
    )

    # Get all refinement_i columns
    refinement_cols = [col for col in df.columns if col.startswith("refinement_")]

    # Get the last non-empty refinement column for each row
    # Stores the name of the refinement column
    last_refinements = df[refinement_cols].apply(
        lambda row: row.last_valid_index(), axis=1
    )

    for idx, pred in enumerate(df["prediction"]):
        # Skip empty predictions
        if str(pred) == "nan":
            continue

        # Extract all facts from answer set (prediction)
        facts_in_as = pred.split("\n")

        # Extract the final ASP, remove spaces as normalization
        program = str(df.loc[idx, last_refinements[idx]]).replace(" ", "")

        # Check if ANY answer set fact appears in the ASP
        if any([True for fact in facts_in_as if fact in program]):
            print(f"Found a match for {idx}\n{program[:100]}...")
            print(f"Facts in answer set: {facts_in_as}")
            print()
