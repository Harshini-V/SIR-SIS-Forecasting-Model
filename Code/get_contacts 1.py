import numpy as np
import pandas as pd  # For handling CSV file creation
from tqdm import tqdm  # Import tqdm for progress tracking

def hhc_expand_dims(hcw_hcw_contact, simulation_period):
    hcw_hcw_contact = np.repeat(np.expand_dims(hcw_hcw_contact, axis=0), simulation_period, axis=0)
    hcw_hcw_contact[6, :, :, :] = 0  # Zero contacts for breaks
    hcw_hcw_contact[13, :, :, :] = 0
    hcw_hcw_contact[20, :, :, :] = 0
    hcw_hcw_contact[27, :, :, :] = 0
    return hcw_hcw_contact

# Use 1 for mrsa, 6 for flu
contact_distance = 1
simulation_period = 30
day = 10  # Day simulated 

npzfile_hcp = np.load(f"contact_data/patient_arrays_day{day}_{contact_distance}ft.npz")
hpc_original = npzfile_hcp["hcw_patient_contact_arrays"]
ppc_original = npzfile_hcp["patient_patient_contact_arrays"]

# Load HCW-HCW arrays
npzfile_hhc = np.load(f"contact_data/hhc_arrays_day{day}_{contact_distance}ft.npz")
hhc_same_chair = npzfile_hhc["hhc_same_chair"]
hhc_adj_chair = npzfile_hhc["hhc_adj_chair"]
hhc_both_center = npzfile_hhc["hhc_both_center"]
hhc_other_places = npzfile_hhc["hhc_other_places"]
hhc_total = npzfile_hhc["hhc_total"]

hhc_same_chair = hhc_expand_dims(hhc_same_chair, simulation_period)
hhc_adj_chair = hhc_expand_dims(hhc_adj_chair, simulation_period)
hhc_both_center = hhc_expand_dims(hhc_both_center, simulation_period)
hhc_other_places = hhc_expand_dims(hhc_other_places, simulation_period)
hhc_total = hhc_expand_dims(hhc_total, simulation_period)

# Create a DataFrame to store results
contact_results = []
sim_period = 30

for day_index in tqdm(range(sim_period), desc="Days"):
    print(contact_distance)
    for timestep in tqdm(range(6822), desc="Time Steps", leave=False):
        timestep_contacts = {
            'Day': day_index + 1,
            'Timestep': timestep,
            'HCP Contacts': [],
            'PPC Contacts': [],
            'HHC Same Chair Contacts': [],
            'HHC Adj Chair Contacts': [],
            'HHC Both Center Contacts': [],
            'HHC Other Places Contacts': [],
            'HHC Total Contacts': []
        }
        
        # Check HCP contact and record the indices of HCWs and patients in contact
        for hcw_idx in range(hpc_original.shape[1]):  # Loop over HCWs
            for patient_idx in range(hpc_original.shape[2]):  # Loop over patients
                if hpc_original[day_index, hcw_idx, patient_idx, timestep] > 0:
                    timestep_contacts['HCP Contacts'].append((hcw_idx, patient_idx))
        
        # Check PPC contact and record the indices of patients in contact
        for patient1_idx in range(ppc_original.shape[1]):  # Loop over the first patient
            for patient2_idx in range(ppc_original.shape[2]):  # Loop over the second patient
                if ppc_original[day_index, patient1_idx, patient2_idx, timestep] > 0:
                    timestep_contacts['PPC Contacts'].append((patient1_idx, patient2_idx))
        
        # Repeat for other contact types (HHC same chair, adj chair, etc.)
        for hhc_type, hhc_array in zip(
            ['HHC Same Chair Contacts', 'HHC Adj Chair Contacts', 'HHC Both Center Contacts', 'HHC Other Places Contacts', 'HHC Total Contacts'],
            [hhc_same_chair, hhc_adj_chair, hhc_both_center, hhc_other_places, hhc_total]
        ):
            for hcw1_idx in range(hhc_array.shape[1]):  # Loop over the first HCW (size 11)
                for hcw2_idx in range(hhc_array.shape[2]):  # Loop over the second HCW (size 11)
                    #print(f"Index out of bounds: hcw1_idx {hcw1_idx}, hcw2_idx {hcw2_idx}")
                    if hhc_array[day_index, hcw1_idx, hcw2_idx, timestep] > 0:
                        timestep_contacts[hhc_type].append((hcw1_idx, hcw2_idx))
        
        # Append the detailed result for this timestep
        contact_results.append(timestep_contacts)

# Convert the detailed results to a DataFrame
df = pd.DataFrame(contact_results)

# Save the detailed contact DataFrame to CSV
output_file = "detailed_contact_results_mrsa.csv"
df.to_csv(output_file, index=False)

print(f"Detailed contact results saved to {output_file}")


