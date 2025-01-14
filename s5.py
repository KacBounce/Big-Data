import csv
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import pandas as pd
import plotly.express as px


def generate_gantt_chart_matplotlib(station_timelines):
    # Create a DataFrame from the station_timelines
    tasks = []
    for station, tasks_list in station_timelines.items():
        for task in tasks_list:
            tasks.append({
                'Machine': station,
                'Product': task['product'],
                'Quantity': task['quantity'],
                'Start Time': task['start_time'],
                'End Time': task['end_time']
            })

    df = pd.DataFrame(tasks)

    # Create a Plotly Gantt chart
    fig = px.timeline(df,
                      x_start="Start Time",
                      x_end="End Time",
                      y="Machine",
                      color="Product",
                      title="Production Process Visualization",
                      labels={"Product": "Product Category",
                              "Machine": "Production Machine"},
                      hover_data=["Quantity"])

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Machine",
        showlegend=True,
        xaxis=dict(
            tickformat="%d.%m.%Y %H:%M",
            tickangle=45
        ),
        height=600
    )

    fig.show()


def read_csv(file_path):
    """Reads a CSV file and returns its rows as dictionaries."""
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter=';')
        return [row for row in reader]

# Parse input files


def parse_inputs():
    with open('ListaANC.csv', mode='r') as file:
        reader = csv.reader(file, delimiter=';')
        # Skip the first column (empty), take the rest as product IDs
        product_ids = next(reader)[1:]

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
                reconfig_times[product][target_product] = int(
                    time) if time.isdigit() else float('inf')

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
        writer.writerow(['Station', 'Total Production Time',
                         'Total Reconfiguration Time', 'Total Time'])
        for station, times in total_times.items():
            total_time = times['production'] + times['reconfig']
            writer.writerow([station, times['production'],
                             times['reconfig'], total_time])

    with open('detailed_schedule.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Station', 'Product', 'Quantity',
                         'Start Time', 'End Time'])
        for station, tasks in station_timelines.items():
            for task in tasks:
                # Use end time without reconfiguration time
                writer.writerow([station, task['product'], task['quantity'], task['start_time'].strftime(
                    '%d.%m.%Y %H:%M:%S'), task['end_time'].strftime('%d.%m.%Y %H:%M:%S')])


def genetic_algorithm(products, product_to_station, reconfig_times, prod_times, demands, start_date, generations=100, population_size=100, elite_size=10, mutation_rate=0.1, crossover_rate=0.8):
    """Genetic algorithm with elitism and optimized mutation rate for scheduling."""
    # Initial population
    population = [generate_schedule(products, demands, product_to_station, start_date, prod_times, reconfig_times)
                  for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_schedule(
            schedule, product_to_station, reconfig_times, prod_times, start_date) for schedule in population]

        # Pair population with fitness scores
        paired_population = list(zip(fitness_scores, population))
        paired_population.sort(key=lambda x: x[0])

        # Elitism: Keep the top N individuals
        elites = [individual for _,
                  individual in paired_population[:elite_size]]

        # Selection: Select remaining individuals based on fitness
        selected = select_population(
            [individual for _, individual in paired_population], fitness_scores)

        # Crossover and Mutation: Generate new population
        new_population = elites.copy()  # Start with elites
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)

            # Apply crossover with a probability determined by crossover_rate
            if random.random() <= crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2  # No crossover, just carry parents over

            # Apply mutation with a chance determined by mutation_rate
            if random.random() <= mutation_rate:
                child1 = mutate(child1)
            if random.random() <= mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        # Ensure population size matches exactly
        population = new_population[:population_size]

    # Return the best schedule from the final generation
    best_schedule = min(population, key=lambda sch: evaluate_schedule(
        sch, product_to_station, reconfig_times, prod_times, start_date))
    return best_schedule


def generate_schedule(products, demands, product_to_station, start_date, prod_times, reconfig_times):
    """Generates an initial schedule with random machine assignments and no strict order."""
    # Sort demands by deadline
    demands.sort(key=lambda x: x['deadline'])

    schedule = []
    machine_end_times = {station: start_date for station in set(
        [station for stations in product_to_station.values() for station in stations])}

    for demand in demands:
        product = demand['product']
        quantity = demand['quantity']
        deadline = demand['deadline']

        # Find the earliest available station
        available_station = None
        earliest_end_time = None
        for station in product_to_station[product]:
            if available_station is None or machine_end_times[station] < earliest_end_time:
                available_station = station
                earliest_end_time = machine_end_times[station]

        # Calculate reconfiguration time if needed
        reconfig_time = 0
        if machine_end_times[available_station] > start_date:
            reconfig_time = reconfig_times.get(
                product, {}).get(product, 0)

        production_time = quantity / prod_times[product]

        # Schedule task
        start_time = max(
            machine_end_times[available_station], start_date) + timedelta(minutes=reconfig_time)
        end_time = start_time + timedelta(minutes=production_time)

        schedule.append({
            'product': product,
            'quantity': quantity,
            'start_time': start_time,
            'end_time': end_time,
            'selected_machine': available_station,
        })

        # Update station end time
        machine_end_times[available_station] = end_time

    return schedule


def evaluate_schedule(schedule, product_to_station, reconfig_times, prod_times, start_date):
    """Evaluates the fitness of a schedule ensuring no gaps between tasks, with machine-specific availability."""
    total_time = 0
    last_end_time = {}  # To track the last end time for each machine
    last_product_on_machine = {}  # To track the last product handled by each machine

    for task in schedule:
        product = task['product']
        quantity = task['quantity']
        selected_machine = task['selected_machine']

        last_machine_end_time = last_end_time.get(selected_machine, start_date)

        # Calculate reconfiguration time
        reconfig_time = 0
        if selected_machine in last_product_on_machine and last_product_on_machine[selected_machine] != product:
            reconfig_time = reconfig_times.get(
                last_product_on_machine[selected_machine], {}).get(product, 0)

        # Calculate production time
        production_time = quantity / prod_times[product]

        task['start_time'] = max(
            last_machine_end_time, start_date) + timedelta(minutes=reconfig_time)
        task['end_time'] = task['start_time'] + \
            timedelta(minutes=production_time)

        last_end_time[selected_machine] = task['end_time']
        last_product_on_machine[selected_machine] = product

        total_time += production_time + reconfig_time

    return total_time


def mutate(schedule):
    task = random.choice(schedule)

    # Mutate by swapping two tasks in the same station
    machine_tasks = [
        t for t in schedule if t['selected_machine'] == task['selected_machine']]
    if len(machine_tasks) > 1:
        swap_task = random.choice(machine_tasks)
        task['start_time'], swap_task['start_time'] = swap_task['start_time'], task['start_time']
        task['end_time'], swap_task['end_time'] = swap_task['end_time'], task['end_time']

    return schedule


def select_population(population, fitness_scores):
    """Selects the top half of the population based on fitness."""
    paired_population = list(zip(fitness_scores, population))
    sorted_population = sorted(paired_population, key=lambda x: x[0])
    return [x[1] for x in sorted_population[:len(sorted_population)//2]]


def crossover(parent1, parent2):
    """Performs single-point crossover between two parents."""
    point = random.randint(0, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Main function


def main():
    products, product_to_station, reconfig_times, prod_times, demands = parse_inputs()
    start_date = datetime.strptime(
        '20.01.2018 08:00:00', '%d.%m.%Y %H:%M:%S')  # Example start date

    best_schedule = genetic_algorithm(
        products, product_to_station, reconfig_times, prod_times, demands, start_date)

    # Convert the best schedule into station timelines
    station_timelines = {}
    total_times = {}
    for task in best_schedule:
        station = task['selected_machine']
        if station not in station_timelines:
            station_timelines[station] = []
            total_times[station] = {'production': 0, 'reconfig': 0}

        end_time = task['start_time'] + \
            timedelta(minutes=task['quantity'] / prod_times[task['product']])
        station_timelines[station].append({
            'product': task['product'],
            'quantity': task['quantity'],
            'start_time': task['start_time'],
            'end_time': end_time
        })
        total_times[station]['production'] += (
            task['quantity'] / prod_times[task['product']])

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
