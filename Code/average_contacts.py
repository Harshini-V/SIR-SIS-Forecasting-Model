import numpy as np
import matplotlib.pyplot as plt

def hhc_expand_dims(hcw_hcw_contact, simulation_period):
    hcw_hcw_contact = np.repeat(np.expand_dims(hcw_hcw_contact, axis=0), simulation_period, axis=0)
    hcw_hcw_contact[6, :, :, :] = 0
    hcw_hcw_contact[13, :, :, :] = 0
    hcw_hcw_contact[20, :, :, :] = 0
    hcw_hcw_contact[27, :, :, :] = 0
    return hcw_hcw_contact

# One for mrsa and 6 for flu
contact_distance = 6
simulation_period = 30
day = 10  # Day used to build the sim data

npzfile_hcp = np.load(f"contact_data/patient_arrays_day{day}_{contact_distance}ft.npz")
hpc_original = npzfile_hcp["hcw_patient_contact_arrays"]
ppc_original = npzfile_hcp["patient_patient_contact_arrays"]

# Calculate timestep totals across all days
hcp_timestep_totals = hpc_original.sum(axis=(1, 2))  # Shape: (30, 6822)
ppc_timestep_totals = ppc_original.sum(axis=(1, 2))  # Shape: (30, 6822)

# Exclude specific days (7, 14, 21, 28) from the totals
exclude_days = {7, 14, 21, 28}
include_days = [i for i in range(30) if i + 1 not in exclude_days]

hcp_timestep_totals_included = hcp_timestep_totals[include_days, :]
ppc_timestep_totals_included = ppc_timestep_totals[include_days, :]

# Calculate averages over the included days
hcp_timestep_avg = hcp_timestep_totals_included.mean(axis=0)
ppc_timestep_avg = ppc_timestep_totals_included.mean(axis=0)

# Plot average contacts per timestep
plt.figure(figsize=(14, 10))

# HCP-Patient Contacts
plt.plot(range(1, len(hcp_timestep_avg) + 1), hcp_timestep_avg, label='HCP-Patient Avg Contacts', color='C9')

# Patient-Patient Contacts
plt.plot(range(1, len(ppc_timestep_avg) + 1), ppc_timestep_avg, label='Patient-Patient Avg Contacts', color='C3')

# Add labels and legend
plt.title('Average MRSA Contacts Per Timestep Over 30 Days', fontsize=16)
plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Average Contacts', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

