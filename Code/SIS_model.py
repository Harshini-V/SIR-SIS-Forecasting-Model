import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
from collections import defaultdict
import random

# Function to create a graph with nodes for patients and HCWs
def create_base_graph(num_patients=40, num_hcws=11):
    graph = nx.Graph()
    for i in range(num_patients):  # Add patients
        graph.add_node(i, type='patient')
    for i in range(num_patients, num_patients + num_hcws):  # Add HCWs
        graph.add_node(i, type='hcw')
    return graph

# Infect susceptible nodes (based on neighbors)
def S_to_I(node, states, graph, beta):
    infected_neighbors = sum(1 for neighbor in graph.neighbors(node) if states[neighbor] == 'I')
    infection_prob = 1 - np.exp(-beta * infected_neighbors)
    if np.random.rand() < infection_prob:
        return 'I'  # Transition to Infected
    return 'S'  # Remain Susceptible

# Recover infected nodes at the end of the day
def recover_infected(states, gamma):
    for node in states:
        if states[node] == 'I' and np.random.rand() < gamma:
            states[node] = 'S'  # Transition back to Susceptible
    return states

def calc_beta(beta_in, duration):
    beta_out = (beta_in * duration) / 75
    return beta_out

# Update states (infection dynamics only for new contacts)
def update_states_infection(states, graph, base_beta, day_data):
    new_states = states.copy()
    # find all the nodes
    for _, row in day_data.iterrows():
        nodes_combined = []
        duration = row['Duration']
        node_1 = row['Node 1']
        node_2 = row['Node 2']
        nodes_combined.append(node_1)
        nodes_combined.append(node_2)
    
        for node in nodes_combined:
            if new_states[node] == 'S':
                beta = calc_beta(duration, base_beta)
                new_states[node] = S_to_I(node, states, graph, beta)
    return new_states 

def SIS_simulation(count_csv, graph, beta_init, gamma, initial_infected):
    df = pd.read_csv(count_csv)

    # Initialize states for all nodes
    states = {node: 'S' for node in graph.nodes}
    for node in initial_infected:
        states[node] = 'I'

    # History to store S, I, R counts over time
    history = []

    # Add Day 0 to history with initial conditions
    history.append({
        'Day': 0,
        'S': sum(1 for s in states.values() if s == 'S'),
        'I': sum(1 for s in states.values() if s == 'I'),
        'R': sum(1 for s in states.values() if s == 'R')
    })

    days = df['Day'].unique()
    #print(days)
    for day in days:
        #print(day)
        day_data = df[df['Day'] == day]
        #print(day_data)
        
        # Clear existing edges for the day (optional, if edges are day-specific)
        graph.clear_edges()
        
        # Add edges based on contacts in the day_data
        for _, row in day_data.iterrows():
            node_1 = row['Node 1']
            node_2 = row['Node 2']
            graph.add_edge(node_1, node_2)  # Add edge between the nodes

        # Update infection states based on the contact data and new beta calculation
        states = update_states_infection(states, graph, beta_init, day_data)

        # Run recovery (I to S) only once at the end of the day
        states = recover_infected(states, gamma)

        # Record state counts
        history.append({
            'Day': day,
            'S': sum(1 for s in states.values() if s == 'S'),
            'I': sum(1 for s in states.values() if s == 'I')
        })
    return history


# Function to run multiple simulations and average the results
def run_multiple_simulations_by_day(num_runs, csv_file, num_patients, num_hcws, beta_init, gamma, initial_infected):
    aggregated_results = defaultdict(lambda: {'S': [], 'I': []})
    
    for i in range(num_runs):
        random_patient_mrsa = random.choice([p for p in unique_patients if p not in initial_infected])
        random_hcw_mrsa = random.choice([h for h in unique_hcw if h not in initial_infected])

        initial_infected.append(random_patient_mrsa)
        initial_infected.append(random_hcw_mrsa)
        print(initial_infected)

        print("Running simulation", i+1)
        # Create a new graph for each run
        G = create_base_graph(num_patients, num_hcws)
        history = SIS_simulation(csv_file, G, beta_init, gamma, initial_infected)
        initial_infected.clear() # clear for the next day
        
        # Aggregate results by day
        for record in history:
            day = record['Day']
            aggregated_results[day]['S'].append(record['S'])
            aggregated_results[day]['I'].append(record['I'])
    
    # Average the results by day over all runs
    averaged_history = []
    for day, records in aggregated_results.items():
        avg_record = {
            'Day': day,
            'S': np.mean(records['S']),
            'I': np.mean(records['I'])
        }
        averaged_history.append(avg_record)
    
    return averaged_history

# Main script for running the SIS model with multiple runs
csv_file = "contact_sequences_mrsa.csv"  # Path to your CSV file
num_patients = 40
num_hcws = 11
beta = 0.05  # Transmission rate
gamma = 0.04  # Daily recovery rate
initial_infected = []  # Example initial infected nodes
unique_patients = [0, 3, 6, 9, 12, 17]
unique_hcw = [42, 43, 44, 45, 46, 47, 48, 49]
num_runs = 50  # Number of runs (can also try 100, 500)

# Run the simulations
averaged_results_by_day = run_multiple_simulations_by_day(num_runs, csv_file, num_patients, num_hcws, beta, gamma, initial_infected)

# Convert averaged results to DataFrame for analysis or visualization
averaged_df_by_day = pd.DataFrame(averaged_results_by_day)

# Plot the averaged results by day
plt.figure(figsize=(12, 6))
plt.plot(averaged_df_by_day['Day'], averaged_df_by_day['S'], label='Susceptible (S)', color='C0')
plt.plot(averaged_df_by_day['Day'], averaged_df_by_day['I'], label='Infected (I)', color='C1')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title(f'SIS Model Simulation (Averaged over {num_runs} Simulations)')
plt.legend()
plt.grid()

plt.savefig(f'SIS_simulation_plot_fin.png', dpi=300, bbox_inches='tight')

plt.show()