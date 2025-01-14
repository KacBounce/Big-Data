import csv
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates

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

def generate_gantt_chart_matplotlib(station_timelines):
    fig, ax = plt.subplots(figsize=(16, 10))  # Increase figure size for better clarity

    # Generate a color map for products
    unique_products = {task['product'] for tasks in station_timelines.values() for task in tasks}
    product_colors = {product: f"C{i}" for i, product in enumerate(unique_products)}  # Use matplotlib default color cycle

    for station, tasks in station_timelines.items():
        for task in tasks:
            start = date2num(task['start_time'])
            end = date2num(task['end_time'])
            duration = task['end_time'] - task['start_time']
            duration_minutes = int(duration.total_seconds() / 60)
            
            # Plot each task as a bar
            ax.barh(
                station, 
                end - start, 
                left=start, 
                color=product_colors[task['product']],
                edgecolor="black"
            )
            
            # Annotate each bar with product and duration
            ax.text(
                start + (end - start) / 2, 
                station, 
                f"{task['product']} ({duration_minutes} min)", 
                ha='center', 
                va='center', 
                fontsize=8, 
                color='white',
                fontweight='bold',
                clip_on=True
            )

    # Add a legend for product colors
    legend_patches = [
        plt.Line2D([0], [0], color=color, lw=4, label=product) 
        for product, color in product_colors.items()
    ]
    ax.legend(
        handles=legend_patches, 
        loc='upper right', 
        fontsize='small', 
        title="Products"
    )

    # Format the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Station")
    ax.set_title("Production Schedule Gantt Chart")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))  # Improved date format
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# Genetic Algorithm Implementation
def genetic_algorithm(products, product_to_station, reconfig_times, prod_times, demands, start_date, generations=1000, population_size=200, elite_size=10):
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

# Helpers for Genetic Algorithm
def generate_schedule(products, demands, start_date):
    """Randomly generates an initial schedule."""
    schedule = []
    for demand in demands:
        product = demand['product']
        quantity = demand['quantity']
        # Prevent too early start
        start_time = max(start_date, demand['deadline'] - timedelta(minutes=quantity * 2))
        schedule.append({'product': product, 'quantity': quantity, 'start_time': start_time})
    schedule.sort(key=lambda task: task['start_time'])  # Sort tasks by start time
    return schedule

def evaluate_schedule(schedule, product_to_station, reconfig_times, prod_times):
    """Evaluates the fitness of a schedule with penalties for overlaps and early starts."""
    total_time = 0
    last_end_time = {}

    for task in schedule:
        product = task['product']
        quantity = task['quantity']
        station = product_to_station[product][0]  # Use first compatible station
        production_time = quantity / prod_times[product]

        # Ensure tasks on the same station don't overlap
        if station in last_end_time and task['start_time'] < last_end_time[station]:
            overlap_penalty = (last_end_time[station] - task['start_time']).total_seconds() / 60
            total_time += overlap_penalty  # Add penalty for overlap
        
        last_end_time[station] = task['start_time'] + timedelta(minutes=production_time)

        # Add reconfiguration time if applicable
        if schedule.index(task) > 0:
            prev_product = schedule[schedule.index(task) - 1]['product']
            total_time += reconfig_times[prev_product].get(product, float('inf'))

        total_time += production_time

    return total_time

def select_population(population, fitness_scores):
    """Selects the top half of the population based on fitness."""
    paired_population = list(zip(fitness_scores, population))
    sorted_population = sorted(paired_population, key=lambda x: x[0])
    return [x[1] for x in sorted_population[:len(sorted_population)//2]]

def crossover(parent1, parent2):
    """Performs single-point crossover between two parents."""
    point = random.randint(0, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(schedule):
    """Randomly mutates a schedule by changing the start time of a random task."""
    task = random.choice(schedule)
    task['start_time'] += timedelta(minutes=random.randint(-60, 60))  # Mutate by up to an hour


# Main function
def main():
    products, product_to_station, reconfig_times, prod_times, demands = parse_inputs()
    start_date = datetime.strptime('01.01.2018 08:00:00', '%d.%m.%Y %H:%M:%S')  # Example start date

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
