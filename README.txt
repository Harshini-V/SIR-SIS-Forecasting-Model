README for SIR-SIS Fusion: MRSA and Influenza Co-infection Forecasting
======================================================================
Dataset Preprocessing Statistics
----------------------------------------------------------------------
File: average_contacts.py, generate_statistics.py
Run: python3 average_contacts.py, python3 generate_statistics.py
Inputs for MRSA: patient_arrays_day10_1ft.npz, hcw_arrays_day10_1ft.npz
Inputs for Flu: patient_arrays_day10_6ft.npz, hcw_arrays_day10_6ft.npz
Output for MRSA: Average_MRSA_Contacts.png, additional_network_statistics_mrsa.csv
Output for Flu: Average_flu_Contacts.png, additional_network_statistics_flu.csv

Run using:
- Python (>= 3.10)
Required packages:
- pandas
- numpy
- graph statistics
- networkx

Before running average_contacts.py or generate_statistics.py, you first need to create the movement data numpy arrays through https://github.com/HankyuJang/Dialysis_COVID19. The github page provides specific instructions on how to download and get their scripts working. The main script to run is prepare_contact_networks.sh, it will run all the scripts necessary to make the contact data. 

When running make sure to change contact distance (1 for mrsa, 6 for flu) and change names of files to reflect results

Data Processing
----------------------------------------------------------------------
File: get_contacts.py, count_interactions_timesteps.py
Run: python3 get_contacts.py, python3 count_interactions_timesteps.py

Inputs for get_contacts MRSA: patient_arrays_day10_1ft.npz, hcw_arrays_day10_1ft.npz
Inputs for get_contacts Flu: patient_arrays_day10_6ft.npz, hcw_arrays_day10_6ft.npz
Outputs: detailed_contact_results_mrsa.csv, detailed_contact_results_flu.csv

Inputs for count_interactions_timesteps.py: detailed_contact_results_mrsa.csv, detailed_contact_results_flu.csv (output from get_contacts.py)
Outputs: contact_sequences_mrsa.csv, contact_sequences_flu.csv

Run using:
- Python (>= 3.10)
Required packages:
- pandas
- numpy
- tqdm
- collections (defaultdict)

get_contacts.py also uses the GitHub to generate the contact arrays, follow the same instructions. Run MRSA and flu separately in the code. This is all that needs to be changed (input and output file names). Output will give csv of each contact found in the contact results csv files, listing nodes involved, timestep start and stop, and duration of interaction.

----------------------------------------------------------------------
SIR-SIS ODE Model
----------------------------------------------------------------------
Baseline SIR and SIS separate models:
SIR MODEL
---------
File: SIR_model.py
Run: python SIR_model.py
Inputs: contact_sequences_flu.csv (output from previous data processing: count_interactions_timesteps.py)
Outputs: SIR_simulation_plot_fin.png

Run using:
- Python (>= 3.10)

Required packages:
- numpy
- pandas
- networkx
- matplotlib.pyplot
- collections (defaultdict)
- random

SIS MODEL
---------
File: SIS_model.py
Run: python SIS_model.py
Inputs: contact_sequences_mrsa.csv (output from previous data processing: count_interactions_timesteps.py)
Outputs: SIS_simulation_plot_fin.png

Run using:
- Python (>= 3.10)

Required packages:
- numpy
- pandas
- networkx
- matplotlib.pyplot
- collections (defaultdict)
- random

SIR-SIS MODEL
---------
File: SIR_SIS_model.py
Run: python SIR_SIS_model.py
Inputs: contact_sequences_flu.csv, contact_sequences_mrsa.csv (output from previous data processing: count_interactions_timesteps.py)
Outputs: full_results.csv, SIR_SIS_simulation_plot.png

Run using:
- Python (>= 3.10)

Required packages:
- numpy
- pandas
- networkx
- matplotlib.pyplot
- collections (defaultdict)
- random

----------------------------------------------------------------------
Machine Learning Code
----------------------------------------------------------------------
Description of Package
----------------------
- ML_code_epi_final.ipynb
This jupyter notebook contains the code for the machine learning implementation.

Installation
------------
Prerequisites:
Before running the code, ensure you have the following installed:
- Python (>= 3.10)
- Jupyter Notebook
- Required Python packages
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - sklearn
    - xgboost

Usage
-----
Running the Jupyter Notebook:
1. Open the Jupyter Notebook environment
2. Open the main notebook file: ML_code_epi_final.ipynb
3. Change the file paths for the two data files
4. Select "Run All" at the top or run the cells sequentially 


