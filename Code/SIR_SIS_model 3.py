import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random

'''
Project: CSE 8803 Co Infection Model
Authors: Lauren Fogel, Asma Khimani, Harshini Vummadi
'''

# Makes graph using networkx package
def create_base_graph(num_patients=40, num_hcws=11):
    graph = nx.Graph()
    for i in range(num_patients):  # Add patients
        graph.add_node(i, type='patient')
    for i in range(num_patients, num_patients + num_hcws):  # Add HCWs
        graph.add_node(i, type='hcw')
    return graph

# Function to calculate recovery rate for co-infected individuals
def gamma_both(gamma_flu, gamma_mrsa, alpha):
    return 1 / (1/gamma_flu + 1/gamma_mrsa + alpha)

# Infect susceptible nodes (based on neighbors) for both flu and MRSA - inital infections
def S_to_I_flu(node, states, graph, beta_flu, beta_mrsa, delta):
    # Count neighbors infected with flu or both
    infected_flu_neighbors = sum(1 for neighbor in graph.neighbors(node) 
                                 if states[neighbor] in ['I_flu', 'I_both'])

    # Infection probabilities for flu
    infection_prob_flu = 1 - np.exp(-beta_flu * infected_flu_neighbors)

    # Randomly determine infection
    if np.random.rand() < infection_prob_flu and states[node] != 'I_mrsa':
        return 'I_flu'  # Transition to flu infection
    
    return states[node]  # No change in state

def S_to_I_mrsa(node, states, graph, beta_flu, beta_mrsa, delta):

    # Count neighbors infected with MRSA or both
    infected_mrsa_neighbors = sum(1 for neighbor in graph.neighbors(node) 
                                  if states[neighbor] in ['I_mrsa', 'I_mrsa_R_flu', 'I_both'])

    # Infection probabilities for MRSA
    infection_prob_mrsa = 1 - np.exp(-beta_mrsa * infected_mrsa_neighbors)

    # Adjust susceptibility to MRSA if already infected with flu
    if states[node] == 'I_flu':
        infection_prob_mrsa *= (1 + delta)

    # Randomly determine infection
    elif np.random.rand() < infection_prob_mrsa and states[node] != 'I_flu':
        return 'I_mrsa'  # Transition to MRSA infection
    
    return states[node]  # No change in state

def S_to_I_both_flu(node, states, graph, beta_flu, beta_mrsa, gamma_both_rate, delta):

    if states[node] == 'I_flu':
        infected_mrsa_neighbors = sum(1 for neighbor in graph.neighbors(node) if states[neighbor] in ['I_mrsa', 'I_mrsa_R_flu', 'I_both'])
        effective_beta_mrsa = beta_mrsa * (1 + delta)
        infection_prob_mrsa = 1 - np.exp(-effective_beta_mrsa * infected_mrsa_neighbors)
        if np.random.rand() < infection_prob_mrsa:
            return 'I_both'  # Transition to Co-infected (both flu and MRSA)
    return states[node]  # Remain in the same state

def S_to_I_both_mrsa(node, states, graph, beta_flu, beta_mrsa, gamma_both_rate, delta):

    if states[node] in ['I_mrsa']:
        infected_flu_neighbors = sum(1 for neighbor in graph.neighbors(node) if states[neighbor] in ['I_flu', 'I_both'])
        infection_prob_flu = 1 - np.exp(-beta_flu * infected_flu_neighbors)
        if np.random.rand() < infection_prob_flu:
            return 'I_both'  # Transition to Co-infected (both flu and MRSA)
    return states[node]  # Remain in the same state

# Transition for MRSA-Recovered Flu state (I_mrsa_R_flu) - infected with MRSA after recovering from flu
def I_mrsa_S_flu_to_R(node, gamma_mrsa):
    if np.random.rand() < gamma_mrsa:
        return 'S_mrsa_R_flu'
    return 'I_mrsa_R_flu'

# Transition for Susceptible MRSA-Recovered Flu state (S_mrsa_R_flu)
def S_mrsa_R_flu_to_I_mrsa(node, states, graph, beta_mrsa):
    infected_mrsa_neighbors = sum(1 for neighbor in graph.neighbors(node) if states[neighbor] in ['I_mrsa', 'I_mrsa_R_flu', 'I_both'])
    infection_prob_mrsa = 1 - np.exp(-beta_mrsa * infected_mrsa_neighbors)
    if np.random.rand() < infection_prob_mrsa:
        return 'I_mrsa_R_flu'  # Transition to MRSA-Recovered Flu state with MRSA infection
    return 'S_mrsa_R_flu'  # Remain in Susceptible state

# Recover flu only nodes
def I_flu_to_R(node, gamma_flu):
    if np.random.rand() < gamma_flu:
        return 'S_mrsa_R_flu'
    return 'I_flu'

# Recover MRSA only nodes - make re-susceptible
def I_mrsa_to_S(node, gamma_mrsa):
    if np.random.rand() < gamma_mrsa:
        return 'S'
    return 'I_mrsa'

# Recover co-infected nodes
def I_both_to_R(node, gamma_both_rate):
    if np.random.rand() < gamma_both_rate:
        return 'S_mrsa_R_flu'  # Recover from flu, re-susceptible to MRSA
    return 'I_both'  # Remain co-infected

def calc_beta(beta_in, duration):
    beta_out = (beta_in * duration) / 75
    return beta_out

# Update states
def update_states_flu(states, graph, beta_flu_init, beta_mrsa_init, gamma_both_rate, delta, day_data):
    new_states = states.copy()  # Copy to keep original intact
    # Find all the nodes
    for _, row in day_data.iterrows():
        nodes_combined = []
        duration = row['Duration']
        node_1 = row['Node 1']
        node_2 = row['Node 2']
        timestep = row['Start']
        nodes_combined.append(node_1)
        nodes_combined.append(node_2)

        for node in nodes_combined:
            # Infect or not with flu
            if new_states[node] == 'S':
                beta_flu = calc_beta(duration, beta_flu_init)
                beta_mrsa = calc_beta(duration, beta_mrsa_init)
                new_states[node] = S_to_I_flu(node, states, graph, beta_flu, beta_mrsa, delta)
            # Try to move to co-infection state
            elif states[node] in ['I_mrsa', 'I_flu', 'I_mrsa_R_flu']:
                beta_flu = calc_beta(duration, beta_flu_init)
                beta_mrsa = calc_beta(duration, beta_mrsa_init)
                new_states[node] = S_to_I_both_flu(node, states, graph, beta_flu, beta_mrsa, gamma_both_rate, delta)
    return new_states

def update_states_mrsa(states, graph, beta_flu_init, beta_mrsa_init, gamma_both_rate, delta, day_data):
    new_states = states.copy()  # Copy to keep original intact
    # Find all the nodes
    for _, row in day_data.iterrows():
        nodes_combined = []
        duration = row['Duration']
        node_1 = row['Node 1']
        node_2 = row['Node 2']
        timestep = row['Start']
        nodes_combined.append(node_1)
        nodes_combined.append(node_2)

        for node in nodes_combined:
            # Infect or not with MRSA
            if states[node] == 'S':
                beta_flu = calc_beta(duration, beta_flu_init)
                beta_mrsa = calc_beta(duration, beta_mrsa_init)
                new_states[node] = S_to_I_mrsa(node, states, graph, beta_flu, beta_mrsa, delta)
            # Try to move to co-infection state
            elif states[node] in ['I_flu', 'I_mrsa']:
                beta_flu = calc_beta(duration, beta_flu_init)
                beta_mrsa = calc_beta(duration, beta_mrsa_init)
                new_states[node] = S_to_I_both_mrsa(node, states, graph, beta_flu, beta_mrsa, gamma_both_rate, delta)
            # Transition from Susceptible MRSA-Recovered Flu state
            if states[node] == 'S_mrsa_R_flu':
                beta_flu = calc_beta(duration, beta_flu_init)
                beta_mrsa = calc_beta(duration, beta_mrsa_init)
                new_states[node] = S_mrsa_R_flu_to_I_mrsa(node, states, graph, beta_mrsa)
    return new_states

def update_states_recover(states, graph, beta_flu_init, beta_mrsa_init, gamma_both_rate, day_data):
    new_states = states.copy()  # Copy to keep original intact
    # Find all the nodes
    for _, row in day_data.iterrows():
        nodes_combined = []
        duration = row['Duration']
        node_1 = row['Node 1']
        node_2 = row['Node 2']
        timestep = row['Start']
        nodes_combined.append(node_1)
        nodes_combined.append(node_2)

        for node in nodes_combined:
            if states[node] == 'I_flu':
                # Recover or not from flu
                new_states[node] = I_flu_to_R(node, gamma_flu)
            elif states[node] == 'I_mrsa':
                # Recover or not from MRSA
                new_states[node] = I_mrsa_to_S(node, gamma_mrsa)
            elif states[node] == 'I_both':
                # Recover or not from co-infection
                new_states[node] = I_both_to_R(node, gamma_both_rate)
            elif states[node] == 'I_mrsa_R_flu':
                # Recover from MRSA (S_mrsa_R_flu)
                new_states[node] = I_mrsa_S_flu_to_R(node, gamma_mrsa)
    return new_states

# Count nodes in S, I, R states
def count_states(states):
    return {state: sum(1 for node in states if states[node] == state) for state in ['S', 'I_flu', 'I_mrsa', 'I_both', 'I_mrsa_R_flu', 'S_mrsa_R_flu']}

def SIR_SIS_simulation(graph, csv_file_flu, csv_file_mrsa, beta_flu, beta_mrsa, gamma_flu, gamma_mrsa, alpha, delta, initial_infected_flu, initial_infected_mrsa):
    df_flu = pd.read_csv(csv_file_flu)
    df_mrsa = pd.read_csv(csv_file_mrsa)  

    states = {node: 'S' for node in graph.nodes}
    for node in initial_infected_flu:
        states[node] = 'I_flu'  # Initial flu-infected nodes
    for node in initial_infected_mrsa:
        states[node] = 'I_mrsa'  # Initial MRSA-infected nodes
        
    # History to store S, I, R counts over time
    history = []

    history.append({
        'Day': 0,
        'S': sum(1 for s in states.values() if s == 'S'),
        'I_flu': sum(1 for s in states.values() if s == 'I_flu'),
        'I_mrsa': sum(1 for s in states.values() if s == 'I_mrsa'),
        'I_both': sum(1 for s in states.values() if s == 'I_both'),
        'I_mrsa_R_flu': sum(1 for s in states.values() if s == 'I_mrsa_R_flu'),
        'S_mrsa_R_flu': sum(1 for s in states.values() if s == 'S_mrsa_R_flu'),
    })

    def combine_states(state_flu, state_mrsa):
        if state_flu == 'I_flu' and state_mrsa == 'I_mrsa':
            return 'I_both' 
        elif state_mrsa == 'I_mrsa':
            return 'I_mrsa'      
        elif state_flu == 'I_flu':
            return 'I_flu'
        elif state_flu == 'I_both' or state_mrsa == 'I_both':
            return 'I_both' 
        elif state_mrsa == 'I_mrsa_R_flu':
            return 'I_mrsa_R_flu'
        elif state_flu == 'S_mrsa_R_flu' or state_mrsa == 'S_mrsa_R_flu':
            return 'S_mrsa_R_flu'
        else:
            return 'S'

    days = df_flu['Day'].unique()
    gamma_both_rate = gamma_both(gamma_flu, gamma_mrsa, alpha)
    for day in range(1, 31):
        #print(day)
        #print(states)
        flu_day_data = df_flu[df_flu['Day'] == day]
        mrsa_day_data = df_mrsa[df_mrsa['Day'] == day]

        # Update graph edges for the current timestep
        graph.clear_edges()

        for _, row in flu_day_data.iterrows():
            node_1 = row['Node 1']
            node_2 = row['Node 2']
            graph.add_edge(node_1, node_2)  # Add edge between the nodes

        states_flu = update_states_flu(states, graph, beta_flu, beta_mrsa, gamma_both_rate, delta, flu_day_data)

        graph.clear_edges()

        for _, row in mrsa_day_data.iterrows():
            node_1 = row['Node 1']
            node_2 = row['Node 2']
            graph.add_edge(node_1, node_2)  # Add edge between the nodes

        states_mrsa = update_states_mrsa(states, graph, beta_flu, beta_mrsa, gamma_both_rate, delta, mrsa_day_data)

        states = {node: combine_states(states_flu[node], states_mrsa[node]) for node in states_flu}

        history.append({
            'Day': day,
            'S': sum(1 for s in states.values() if s == 'S'),
            'I_flu': sum(1 for s in states.values() if s == 'I_flu'),
            'I_mrsa': sum(1 for s in states.values() if s == 'I_mrsa'),
            'I_both': sum(1 for s in states.values() if s == 'I_both'),
            'I_mrsa_R_flu': sum(1 for s in states.values() if s == 'I_mrsa_R_flu'),
            'S_mrsa_R_flu': sum(1 for s in states.values() if s == 'S_mrsa_R_flu')
        })

        # run recover once at the end of day
        states = update_states_recover(states, graph, gamma_flu, gamma_mrsa, gamma_both_rate, flu_day_data)

    return history


# Run Simulation

def run_multiple_simulations_by_day(num_runs, csv_file_flu, csv_file_mrsa, num_patients, num_hcws, beta_flu, beta_mrsa, gamma_flu, gamma_mrsa, alpha, delta, initial_infected_flu, initial_infected_mrsa):
    aggregated_results = defaultdict(lambda: {'S': [], 'I_flu': [], 'I_mrsa': [], 'I_both': [], 'I_mrsa_R_flu': [], 'S_mrsa_R_flu': []})
    full_results = []
    contact_num = 10

    for sim_num in range(num_runs):
        print(f"Running simulation {sim_num + 1}...")
        random_patient_flu = random.choice([p for p in unique_patients if p not in initial_infected_flu and p not in initial_infected_mrsa])
        random_hcw_flu = random.choice([h for h in unique_hcw if h not in initial_infected_flu and h not in initial_infected_mrsa])

        initial_infected_flu.append(random_patient_flu)
        initial_infected_flu.append(random_hcw_flu)

        random_patient_mrsa = random.choice([p for p in unique_patients if p not in initial_infected_flu and p not in initial_infected_mrsa])
        random_hcw_mrsa = random.choice([h for h in unique_hcw if h not in initial_infected_flu and h not in initial_infected_mrsa])

        initial_infected_mrsa.append(random_patient_mrsa)
        initial_infected_mrsa.append(random_hcw_mrsa)
        
        # Create a new graph for each run
        G = create_base_graph(num_patients, num_hcws)

        # Run the simulation
        history = SIR_SIS_simulation(
            G, csv_file_flu, csv_file_mrsa, beta_flu, beta_mrsa, 
            gamma_flu, gamma_mrsa, alpha, delta, initial_infected_flu, initial_infected_mrsa
        )

        initial_infected_flu.clear()
        initial_infected_mrsa.clear()

        # Aggregate results by day
        for record in history:
            day = record['Day']

            # Add to aggregated results
            aggregated_results[day]['S'].append(record['S'])
            aggregated_results[day]['I_flu'].append(record['I_flu'])
            aggregated_results[day]['I_mrsa'].append(record['I_mrsa'])
            aggregated_results[day]['I_both'].append(record['I_both'])
            aggregated_results[day]['I_mrsa_R_flu'].append(record['I_mrsa_R_flu'])
            aggregated_results[day]['S_mrsa_R_flu'].append(record['S_mrsa_R_flu'])

            # Append full details for this simulation
            full_results.append({
                'Simulation': sim_num + 1,
                'Beta_flu': beta_flu,
                'Beta_mrsa': beta_mrsa,
                'Delta': delta,
                'Initial_infected_flu': len(initial_infected_flu),
                'Initial_infected_mrsa': len(initial_infected_mrsa),
                'Contact Number': contact_num,
                **record
            })
    # Average the results by day over all runs
    averaged_history = []
    for day, records in aggregated_results.items():
        avg_record = {
            'Day': day,
            'S': np.mean(records['S']),
            'I_flu': np.mean(records['I_flu']),
            'I_mrsa': np.mean(records['I_mrsa']),
            'I_both': np.mean(records['I_both']),
            'I_mrsa_R_flu': np.mean(records['I_mrsa_R_flu']),
            'S_mrsa_R_flu': np.mean(records['S_mrsa_R_flu'])
        }
        averaged_history.append(avg_record)
    return averaged_history, full_results

# Params
beta_flu=0.3 # infection rate for flu
beta_mrsa=0.05 # infection rate for MRSA
gamma_flu=0.05 # recovery rate for flu
gamma_mrsa=0.04 # recovery rate for MRSA
alpha=0.1 # co-infection dynamic penalty
delta=0.6 # increased susceptibility to MRSA due to current flu infection

csv_file_flu = "contact_sequences_flu.csv"
csv_file_mrsa = "contact_sequences_mrsa.csv"
num_patients = 40
num_hcws = 11 
unique_patients = [0, 3, 6, 9, 12, 17]
unique_hcw = [42, 43, 44, 45, 46, 47, 48, 49]
num_runs = 50

initial_infected_flu = []
initial_infected_mrsa = []

averaged_results_by_day, full_results = run_multiple_simulations_by_day(num_runs, csv_file_flu, csv_file_mrsa, num_patients, num_hcws, beta_flu, beta_mrsa, gamma_flu, gamma_mrsa, alpha, delta, initial_infected_flu, initial_infected_mrsa)
averaged_df_by_day = pd.DataFrame(averaged_results_by_day)
full_results_df = pd.DataFrame(full_results)
full_results_df.to_csv(f'full_results.csv', index=False)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(averaged_df_by_day['Day'], averaged_df_by_day['S'], label='Susceptible (S)', color='C0')
plt.plot(averaged_df_by_day['Day'], averaged_df_by_day['I_flu'], label='Infected Flu (I_flu)', color='red')
plt.plot(averaged_df_by_day['Day'], averaged_df_by_day['I_mrsa'], label='Infected MRSA (I_mrsa)', color='C1')
plt.plot(averaged_df_by_day['Day'], averaged_df_by_day['I_both'], label='Infected Both (I_both)', color='C2')
plt.plot(averaged_df_by_day['Day'], averaged_df_by_day['I_mrsa_R_flu'], label='Infected MRSA Recovered Flu (I_mrsa_R_flu)', color='C6')
plt.plot(averaged_df_by_day['Day'], averaged_df_by_day['S_mrsa_R_flu'], label='Susceptible MRSA Recovered Flu (S_mrsa_R_flu)', color='brown')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title(f'SIR-SIS Model Only Patients (Averaged over {num_runs} Simulations)')
plt.legend()
plt.grid()

plt.savefig(f'SIR_SIS_simulation_plot.png', dpi=300, bbox_inches='tight')

plt.show()
