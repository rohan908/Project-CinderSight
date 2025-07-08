import pandas as pd

# Load the first 100 rows of the CSV file
df = pd.read_csv('data/covariates/Firegrowth_pts_v1_1_2023.csv', nrows=100)

def get_varying_cols(
        df: pd.DataFrame,
        group_by: list[str]):
    
    varying_cols = []
    for col in df.columns:
        if df.groupby(group_by)[col].nunique().max() > 1:
            varying_cols.append(col)
    return varying_cols

vary_by_pixel = get_varying_cols(df, ['ID', 'fireday'])
vary_by_fireday = get_varying_cols(df, ['ID'])
vary_by_fireday_and_not_by_pixel = [col for col in vary_by_fireday if col not in vary_by_pixel]
vary_by_fire_and_not_by_fireday = [col for col in df.columns if col not in vary_by_fireday]

print("Columns that vary by pixel:", vary_by_pixel)
print("\nColumns that vary by fire and fireday only:", vary_by_fireday_and_not_by_pixel)
print("\nColumns that vary by fire only:", vary_by_fire_and_not_by_fireday)

# Calculate and print the ratio of unique values to number of rows for each column
print("\nInverse variation for each per-pixel column (lower is better):")
n_rows = len(df)
for col in vary_by_pixel:
    if col in ['lat', 'lon']:
        continue

    unique_count = df[col].nunique(dropna=False)
    ratio = n_rows / unique_count
    print(f"{col}: {ratio:.3f}")


