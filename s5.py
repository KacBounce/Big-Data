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
                writer.writerow([station, task['product'], task['quantity'], task['start_time'].strftime(
                    '%d.%m.%Y %H:%M:%S'), task['end_time'].strftime('%d.%m.%Y %H:%M:%S')])


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


def genetic_algorithm(products, product_to_station, reconfig_times, prod_times, demands, start_date, generations=1000, population_size=1000, elite_size=10, mutation_rate=0.1, crossover_rate=0.8):
    """Genetic algorithm with elitism and optimized mutation rate for scheduling."""
    # Initial population
    population = [generate_schedule(products, demands, start_date)
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



def generate_schedule(products, demands, start_date):
    """Randomly generates an initial schedule with minimized gaps."""
    schedule = []
    for demand in demands:
        product = demand['product']
        quantity = demand['quantity']

        # Assign tasks as early as possible within their deadlines
        deadline = demand['deadline']
        start_time = max(start_date, deadline -
                         timedelta(minutes=random.randint(quantity, quantity * 2)))
        schedule.append(
            {'product': product, 'quantity': quantity, 'start_time': start_time})

    # Sort all tasks by their start times globally
    schedule.sort(key=lambda task: task['start_time'])
    return schedule


def evaluate_schedule(schedule, product_to_station, reconfig_times, prod_times, start_date):
    """Evaluates the fitness of a schedule ensuring no gaps between tasks, with machine-specific availability."""
    total_time = 0
    last_end_time = {}  # To track the last end time for each machine
    last_product_on_machine = {}  # To track the last product handled by each machine
    first_task_on_machine = {}  # To track if a machine has had its first task scheduled

    for task in schedule:
        product = task['product']
        quantity = task['quantity']

        # Calculate production time based on quantity and production speed
        production_time = quantity / prod_times[product]

        # Find the earliest available machine and start time
        earliest_start_time = datetime.max
        selected_machine = None

        for possible_machine in product_to_station[product]:
            # Get the last end time for the machine (or task start time if it's free)
            last_machine_end_time = last_end_time.get(
                possible_machine, task['start_time'])

            # If it's the first task on this machine, force the start time to be `start_date`
            if possible_machine not in first_task_on_machine:
                last_machine_end_time = start_date
                # Mark as having scheduled the first task
                first_task_on_machine[possible_machine] = True

            # If the machine was previously used for a different product, add reconfiguration time
            if possible_machine in last_product_on_machine and last_product_on_machine[possible_machine] != product:
                # Add reconfiguration time from the previous product to the current one
                reconfig_time = reconfig_times.get(
                    last_product_on_machine[possible_machine], {}).get(product, 0)
                last_machine_end_time += timedelta(minutes=reconfig_time)

            # Update to find the earliest possible machine
            if last_machine_end_time < earliest_start_time:
                earliest_start_time = last_machine_end_time
                selected_machine = possible_machine

        # Assign the task to the selected machine and set its start time
        task['start_time'] = earliest_start_time
        end_time = earliest_start_time + timedelta(minutes=production_time)
        # Update the machine's last end time
        last_end_time[selected_machine] = end_time
        # Update last product on this machine
        last_product_on_machine[selected_machine] = product

        # Update the total time
        total_time += production_time

    return total_time






def select_population(population, fitness_scores):
    """Selects the top half of the population based on fitness."""
    # Pair fitness scores with population and sort by fitness score
    paired_population = list(zip(fitness_scores, population))
    # Sort by fitness score (first element)
    sorted_population = sorted(paired_population, key=lambda x: x[0])
    # Return schedules (second element)
    return [x[1] for x in sorted_population[:len(sorted_population)//2]]


def crossover(parent1, parent2):
    """Performs single-point crossover between two parents."""
    point = random.randint(0, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


def mutate(schedule):
    """Randomly mutates a schedule by changing the start time of a random task."""
    task = random.choice(schedule)
    # Mutate by up to an hour
    task['start_time'] += timedelta(minutes=random.randint(-60, 60))
    return schedule

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
        # Use first compatible station
        station = product_to_station[task['product']][0]
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
