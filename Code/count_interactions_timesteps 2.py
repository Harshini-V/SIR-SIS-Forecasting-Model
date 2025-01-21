import pandas as pd
from collections import defaultdict

# CSV
df = pd.read_csv('detailed_contact_results_mrsa.csv')
#df = pd.read_csv('detailed_contact_results_flu.csv')

# Init dictionary to track pair contacts and timesteps
contact_tracker = defaultdict(lambda: defaultdict(list))

# Function to parse and update contact tracker (hcw)
def track_contacts_hcw(day, contact_list_column, timestep_column):
    for idx, contact_list in contact_list_column.items():
            contacts = eval(contact_list)
            timestep = timestep_column[idx]
            for pair in contacts:
                hcw, patient = pair
                hcw_update = hcw + 40 # fix the number
                pair = hcw_update, patient
                contact_tracker[day][pair].append(timestep)

# Function to parse and update contact tracker (patient)
def track_contacts(day, contact_list_column, timestep_column):
    for idx, contact_list in contact_list_column.items():
            contacts = eval(contact_list) 
            timestep = timestep_column[idx]
            for pair in contacts:
                contact_tracker[day][pair].append(timestep)

# Track contacts for all days in the dataset
for day in df['Day'].unique():
    day_df = df[df['Day'] == day]
    track_contacts_hcw(day, day_df['HCP Contacts'], day_df['Timestep'])
    track_contacts(day, day_df['PPC Contacts'], day_df['Timestep'])

# Helper function to find consecutive groups
def find_consecutive_timesteps(timesteps):
    timesteps.sort()  # Ensure sorted timesteps
    sequences = []
    start = timesteps[0]
    prev = timesteps[0]

    for t in timesteps[1:]:
        if t == prev + 1:  # Consecutive timestep
            prev = t
        else:
            sequences.append((start, prev))
            start = t
            prev = t
    sequences.append((start, prev))  # Add the last sequence
    return sequences

# Collect results into a list for creating a DataFrame
results = []

for day, pairs in contact_tracker.items():
    for pair, timesteps in pairs.items():
        if timesteps:
            sequences = find_consecutive_timesteps(timesteps)
            for start, end in sequences:
                results.append({
                    'Day': day,
                    'Node 1': pair[0],
                    'Node 2': pair[1],
                    'Start': start,
                    'End': end,
                    'Duration': end - start + 1
                })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_csv = "contact_sequences_mrsa.csv"
results_df.to_csv(output_csv, index=False)

print(f"Results have been saved to {output_csv}")
