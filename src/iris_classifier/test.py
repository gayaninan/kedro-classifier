import pandas as pd
import numpy as np
import os
from datetime import datetime

# Step 1: Create a sample DataFrame
data = {
    'prod_id': np.random.randint(1000, 5000, size=100),
    'price': np.random.uniform(10.0, 500.0, size=100),
    'date_of_purchase': pd.date_range(start='2023-01-01', periods=100, freq='D')
}

df = pd.DataFrame(data)

# Step 2: Define the output directory for partitioned Parquet files
output_base_dir = 'partitioned_data'

# Step 3: Generate a timestamp partition name using the current date and time, with only 3 digits for milliseconds
def get_partition_name():
    # Get the current timestamp
    now = datetime.now()
    # Format the timestamp to include only 3 digits for milliseconds
    return now.strftime('%Y-%m-%dT%H.%M.%S.%f')[:-3] + 'Z'

# Step 4: Save the DataFrame as partitioned Parquet files
# This will save the whole dataset under a partition directory based on the current time
partition_name = get_partition_name()

# Define the folder path for this partition based on the current timestamp
partition_dir = os.path.join(output_base_dir, partition_name)

# Ensure the partition directory exists
if not os.path.exists(partition_dir):
    os.makedirs(partition_dir)

# Save the entire DataFrame to a Parquet file named 'test_df.parquet' inside the folder
partition_file_path = os.path.join(partition_dir, 'test_df.parquet')
df.to_parquet(partition_file_path, index=False)

print(f"Data saved in '{partition_dir}' as 'test_df.parquet'.")
