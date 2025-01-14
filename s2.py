import csv
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import pandas as pd
import plotly.express as px

def read_csv(file_path):
    """Reads a CSV file and returns its rows as dictionaries."""
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter=';')
        return [row for row in reader]

# Parse input files
def parse_inputs():
    with open('ListaANC.csv', mode='r') as file:
        reader = csv.reader(file, delimiter=';')
        product_ids = next(reader)[1:]  # Skip the first column (empty), take the rest as product IDs
    
    products = product_ids

    product_to_station = {}
    for row in read_csv('PrasyPrzypisanie.csv'):
        product = row['ANC']
        stations = [int(digit) for digit in row['PRASA']]
        product_to_station[product] = stations
    
    reconfig_times = {}
    with open('setupx.csv', mode='r') as file:
        reader = csv.reader(file, delimiter=';')
        headers = next(reader)
        for row in reader:
            product = row[0]
            reconfig_times[product] = {}
            for i, time in enumerate(row[1:]):
                target_product = headers[i]
                reconfig_times[product][target_product] = int(time) if time.isdigit() else float('inf')
    
    prod_times = {}
    prod_data = read_csv('ListaANCPelna.csv')
    for row in prod_data:
        product = row['Numer ANC komponentu']
        speed = int(row['Predkosc'])
        prod_times[product] = speed

    demands = []
    for row in read_csv('batch-dt.csv'):
        demands.append({
            'product': row['Kod'],
            'quantity': int(row['szt']),
            'deadline': datetime.strptime(row['Termin'], '%d.%m.%Y %H:%M:%S')
        })
    
    return products, product_to_station, reconfig_times, prod_times, demands

# Output functions
def generate_csv_outputs(station_timelines, total_times):
    with open('summary_output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Station', 'Total Production Time', 'Total Reconfiguration Time', 'Total Time'])
        for station, times in total_times.items():
            total_time = times['production'] + times['reconfig']
            writer.writerow([station, times['production'], times['reconfig'], total_time])
    
    with open('detailed_schedule.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Station', 'Product', 'Quantity', 'Start Time', 'End Time'])
        for station, tasks in station_timelines.items():
            for task in tasks:
                writer.writerow([station, task['product'], task['quantity'], task['start_time'].strftime('%d.%m.%Y %H:%M:%S'), task['end_time'].strftime('%d.%m.%Y %H:%M:%S')])

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def generate_gantt_chart_matplotlib(station_timelines):
    # Read data from CSV file into a pandas DataFrame
    df = pd.read_csv('detailed_schedule.csv')
    # Convert 'Start Time' and 'End Time' to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%d.%m.%Y %H:%M:%S')
    df['End Time'] = pd.to_datetime(df['End Time'], format='%d.%m.%Y %H:%M:%S')

    # Calculate duration
    df['Duration'] = df['End Time'] - df['Start Time']

    # Group by Station and Product, to reduce the number of unique rows
    df_grouped = df.groupby(['Station', 'Product']).agg({'Start Time': 'min', 'End Time': 'max', 'Quantity': 'sum'}).reset_index()

    # Create a Plotly Gantt chart
    fig = px.timeline(df_grouped, 
                    x_start="Start Time", 
                    x_end="End Time", 
                    y="Station", 
                    color="Product", 
                    title="Production Process Visualization",
                    labels={"Product": "Product Category", "Station": "Production Station"},
                    hover_data=["Quantity"])

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Production Station",
        showlegend=True,
        xaxis=dict(
            tickformat="%d-%m %H:%M",
            tickangle=45
        ),
        height=600
    )

    fig.show()


# Genetic Algorithm Implementation
def genetic_algorithm(products, product_to_station, reconfig_times, prod_times, demands, start_date, generations=100, population_size=100, elite_size=100):
    """Genetic algorithm with elitism for scheduling optimization."""
    # Initial population
    population = [generate_schedule(products, demands, start_date) for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_schedule(schedule, product_to_station, reconfig_times, prod_times) for schedule in population]

        # Pair population with fitness scores
        paired_population = list(zip(fitness_scores, population))
        paired_population.sort(key=lambda x: x[0])  # Sort by fitness score (lower is better)

        # Elitism: Keep the top N individuals
        elites = [individual for _, individual in paired_population[:elite_size]]

        # Selection: Select remaining individuals based on fitness
        selected = select_population([individual for _, individual in paired_population], fitness_scores)

        # Crossover and Mutation: Generate new population
        new_population = elites.copy()  # Start with elites
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        # Ensure population size matches exactly
        population = new_population[:population_size]

    # Return the best schedule from the final generation
    best_schedule = min(population, key=lambda sch: evaluate_schedule(sch, product_to_station, reconfig_times, prod_times))
    return best_schedule

def generate_schedule(products, demands, start_date):
    """Randomly generates an initial schedule."""
    schedule = []
    for demand in demands:
        product = demand['product']
        quantity = demand['quantity']
        # Start at least after the start_date, and avoid unrealistic gaps
        start_time = max(start_date, demand['deadline'] - timedelta(minutes=random.randint(quantity,quantity * 2)))
        schedule.append({'product': product, 'quantity': quantity, 'start_time': start_time})
    schedule.sort(key=lambda task: task['start_time'])
    return schedule

def evaluate_schedule(schedule, product_to_station, reconfig_times, prod_times):
    """Evaluates the fitness of a schedule with penalties for overlaps and early starts."""
    total_time = 0
    last_end_time = {}  # To track the last end time for each station

    for task in schedule:
        product = task['product']
        quantity = task['quantity']
        
        # Calculate production time based on quantity and production speed
        production_time = quantity / prod_times[product]  # Assuming prod_times gives time per unit

        # Try to find an available station that can accommodate the task at the start time
        station = None
        earliest_start_time = task['start_time']
        
        for possible_station in product_to_station[product]:
            # Check if the station is available at the scheduled start time
            if possible_station not in last_end_time or task['start_time'] >= last_end_time[possible_station]:
                station = possible_station
                break
            else:
                # If the station is busy, track when it will be free
                earliest_start_time = max(earliest_start_time, last_end_time[possible_station])

        # If no station was found initially, the task must wait for the earliest available station
        if station is None:
            station = product_to_station[product][0]  # Pick any station (since they will all be available after waiting)
        
        # Update the task's start time to the earliest available time for the station
        task['start_time'] = earliest_start_time

        # Ensure tasks on the same station don't overlap
        if station in last_end_time and task['start_time'] < last_end_time[station]:
            overlap_penalty = (last_end_time[station] - task['start_time']).total_seconds() / 60
            total_time += overlap_penalty  # Add penalty for overlap
        
        # Update the last end time for the selected station
        last_end_time[station] = task['start_time'] + timedelta(minutes=production_time)

        # Add reconfiguration time if applicable (between consecutive tasks)
        if schedule.index(task) > 0:
            prev_product = schedule[schedule.index(task) - 1]['product']
            total_time += reconfig_times[prev_product].get(product, float('inf'))  # Add reconfiguration time

        # Add the production time for the current task
        total_time += production_time

    return total_time





def select_population(population, fitness_scores):
    """Selects the top half of the population based on fitness."""
    # Pair fitness scores with population and sort by fitness score
    paired_population = list(zip(fitness_scores, population))
    sorted_population = sorted(paired_population, key=lambda x: x[0])  # Sort by fitness score (first element)
    return [x[1] for x in sorted_population[:len(sorted_population)//2]]  # Return schedules (second element)


def crossover(parent1, parent2):
    """Performs single-point crossover between two parents."""
    point = random.randint(0, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(schedule):
    """Randomly mutates a schedule by changing the start time of a random task."""
    task = random.choice(schedule)
    task['start_time'] += timedelta(minutes=random.randint(-60, 60))  # Mutate by up to an hour
    return schedule

# Main function
def main():
    products, product_to_station, reconfig_times, prod_times, demands = parse_inputs()
    start_date = datetime.strptime('20.01.2018 08:00:00', '%d.%m.%Y %H:%M:%S')  # Example start date

    best_schedule = genetic_algorithm(products, product_to_station, reconfig_times, prod_times, demands, start_date)

    # Convert the best schedule into station timelines
    station_timelines = {}
    total_times = {}
    for task in best_schedule:
        station = product_to_station[task['product']][0]  # Use first compatible station
        if station not in station_timelines:
            station_timelines[station] = []
            total_times[station] = {'production': 0, 'reconfig': 0}
        
        end_time = task['start_time'] + timedelta(minutes=task['quantity'] / prod_times[task['product']])
        station_timelines[station].append({
            'product': task['product'],
            'quantity': task['quantity'],
            'start_time': task['start_time'],
            'end_time': end_time
        })
        total_times[station]['production'] += (task['quantity'] / prod_times[task['product']])

        # Add reconfiguration time if applicable
        if len(station_timelines[station]) > 1:
            prev_task = station_timelines[station][-2]
            prev_product = prev_task['product']
            total_times[station]['reconfig'] += reconfig_times[prev_product][task['product']]

    # Generate outputs
    generate_csv_outputs(station_timelines, total_times)
    generate_gantt_chart_matplotlib(station_timelines)

if __name__ == '__main__':
    main()