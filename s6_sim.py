import csv
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import pandas as pd
import plotly.express as px
import math


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


def generate_schedule(products, demands, start_date, product_to_station):
    """Randomly generates an initial schedule with minimized gaps."""
    schedule = []
    machine_first_tasks = {}  # To track the first task for each machine

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

    # Ensure that the first task for each machine starts at start_date
    for task in schedule:
        product = task['product']

        # Get the machine assignment for this task
        assigned_machine = product_to_station[task['product']][0]

        # If this is the first task for this machine, set the start time to start_date
        if assigned_machine not in machine_first_tasks:
            # Set first task's start time to the start_date
            task['start_time'] = start_date
            # Track the first task
            machine_first_tasks[assigned_machine] = task

    return schedule



def evaluate_schedule(schedule, product_to_station, reconfig_times, prod_times):
    """Evaluates the fitness of a schedule ensuring no gaps between tasks, with machine-specific availability."""
    total_time = 0
    last_end_time = {}  # To track the last end time for each machine
    last_product_on_machine = {}  # To track the last product handled by each machine

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


def swap_tasks(schedule, product_to_station, reconfig_times, prod_times):
    """Randomly swap two tasks in the schedule and ensure no gaps."""
    idx1, idx2 = random.sample(range(len(schedule)), 2)

    # Ensure both tasks are swapped properly
    task1 = schedule[idx1]
    task2 = schedule[idx2]

    # Reassign tasks to machines and compute start times again
    machine1 = select_machine_for_task(
        task1, product_to_station, schedule, reconfig_times, prod_times)
    machine2 = select_machine_for_task(
        task2, product_to_station, schedule, reconfig_times, prod_times)

    task1['machine'] = machine1
    task2['machine'] = machine2

    # Recalculate start times and ensure no large gaps
    task1['start_time'] = calculate_start_time(
        task1, schedule, machine1, reconfig_times, prod_times)
    task2['start_time'] = calculate_start_time(
        task2, schedule, machine2, reconfig_times, prod_times)

    # Sort tasks by start time to maintain sequence
    schedule.sort(key=lambda task: task['start_time'])

    return schedule


def select_machine_for_task(task, product_to_station, schedule, reconfig_times, prod_times):
    """Selects a machine for the task based on availability and minimizing gaps."""
    earliest_start_time = datetime.max
    selected_machine = None

    for possible_machine in product_to_station[task['product']]:
        last_end_time = get_last_end_time_for_machine(
            possible_machine, schedule, prod_times)

        # Handle the case where there is no previous task on the machine
        if last_end_time is None:
            # If no previous task, start from the task's initial start time
            last_end_time = task['start_time']
        else:
            # If the machine was previously used for a different product, consider reconfiguration time
            if task.get('product') != get_last_product_for_machine(possible_machine, schedule):
                reconfig_time = reconfig_times.get(
                    get_last_product_for_machine(possible_machine, schedule), {}).get(task['product'], 0)
                last_end_time += timedelta(minutes=reconfig_time)

        # Find the earliest available machine
        if last_end_time < earliest_start_time:
            earliest_start_time = last_end_time
            selected_machine = possible_machine

    return selected_machine


def get_last_product_for_machine(machine, schedule):
    """Returns the last product handled by a given machine."""
    for task in reversed(schedule):
        if task.get('machine') == machine:
            return task['product']
    return None  # Return None if no tasks are assigned to the machine yet


def get_last_end_time_for_machine(machine, schedule, prod_times):
    """Returns the last end time for a given machine."""
    last_end_time = None
    for task in schedule:
        if task.get('machine') == machine:
            last_end_time = task['start_time'] + \
                timedelta(minutes=task['quantity'] /
                          prod_times[task['product']])
    return last_end_time


def calculate_start_time(task, schedule, machine, reconfig_times, prod_times):
    """Calculate the start time for a task based on machine availability and reconfig times."""
    last_end_time = get_last_end_time_for_machine(
        machine, schedule, prod_times)

    # If the machine was previously used for a different product, consider reconfiguration time
    if last_end_time:
        reconfig_time = reconfig_times.get(
            task['product'], {}).get(task['product'], 0)
        last_end_time += timedelta(minutes=reconfig_time)

    # Start time is the maximum of the last end time and the task's original start time
    start_time = max(last_end_time, task['start_time'])

    return start_time


def simulated_annealing(products, product_to_station, reconfig_times, prod_times, demands, start_date, initial_temperature=10000, cooling_rate=0.999, max_iterations=1000):
    """Simulated Annealing for optimizing the production schedule."""
    current_schedule = generate_schedule(
        products, demands, start_date, product_to_station)
    current_cost = evaluate_schedule(
        current_schedule, product_to_station, reconfig_times, prod_times)

    best_schedule = current_schedule
    best_cost = current_cost

    temperature = initial_temperature
    for iteration in range(max_iterations):
        # Generate a neighboring solution using swap_tasks
        neighbor_schedule = swap_tasks(current_schedule.copy(
        ), product_to_station, reconfig_times, prod_times)
        neighbor_cost = evaluate_schedule(
            neighbor_schedule, product_to_station, reconfig_times, prod_times)

        # Decide whether to accept the neighbor
        if neighbor_cost < current_cost:
            current_schedule = neighbor_schedule
            current_cost = neighbor_cost
        else:
            acceptance_probability = math.exp(
                (current_cost - neighbor_cost) / temperature)
            if random.random() < acceptance_probability:
                current_schedule = neighbor_schedule
                current_cost = neighbor_cost

        # Update best solution
        if current_cost < best_cost:
            best_schedule = current_schedule
            best_cost = current_cost

        # Decrease temperature
        temperature *= cooling_rate

    return best_schedule


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


# Main function


def main():
    products, product_to_station, reconfig_times, prod_times, demands = parse_inputs()
    start_date = datetime.strptime(
        '20.01.2018 08:00:00', '%d.%m.%Y %H:%M:%S')  # Example start date

    best_schedule = simulated_annealing(
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
