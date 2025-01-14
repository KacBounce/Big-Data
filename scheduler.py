import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import pandas as pd
import plotly.express as px

def read_csv(file_name):
    """Reads a CSV file and returns its content as a list of dictionaries."""
    with open(file_name, mode='r') as file:
        return list(csv.DictReader(file, delimiter=";"))

def parse_inputs():
    """Reads and parses all input CSV files into structured data."""
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

def calculate_task_times(last_task, product, quantity, reconfig_times, prod_times, deadline=None):
    """Calculates the reconfiguration and production times, considering deadlines."""
    
    last_product = last_task['product'] if last_task else None
    reconfig_time = timedelta(minutes=reconfig_times[last_product][product]) if last_product and last_product != product else timedelta(0)
    production_time = quantity / prod_times[product]
    total_time = reconfig_time + production_time
    
    if deadline:
        time_remaining = deadline - (last_task['end_time'] if last_task else datetime.now())
        if total_time > time_remaining:
            total_time = time_remaining
            reconfig_time = timedelta(0)
            production_time = total_time
    
    return reconfig_time, production_time, total_time

def allocate_tasks_to_stations(demands, product_to_station, station_timelines, reconfig_times, prod_times, start_date):
    """Allocates tasks to stations while considering deadlines and station availability."""
    
    parallel_tasks = []
    station_end_times = {station: start_date for stations in product_to_station.values() for station in stations}
    
    # Sort demands by deadlines to prioritize tasks with earlier deadlines
    demands.sort(key=lambda x: x['deadline'])
    
    for demand in demands:
        product = demand['product']
        quantity = demand['quantity']
        deadline = demand['deadline']
        
        assigned = False
        
        # Loop through all stations that can handle the product
        for station in product_to_station[product]:
            last_task = station_timelines[station][-1] if station_timelines[station] else None
            
            reconfig_time, production_time, total_time = calculate_task_times(last_task, product, quantity, reconfig_times, prod_times)
            
            # Try to start task at the earliest possible time
            start_time = max(station_end_times[station], last_task['end_time'] if last_task else start_date)
            end_time = start_time + total_time
            
            # Debug: Print task start and end times
            print(f"Station {station} - Product {product} - Start: {start_time}, End: {end_time}")
            
            # Check if the station is available at the desired time (no overlap)
            available = True
            for task in station_timelines[station]:
                if (start_time < task['end_time']) and (end_time > task['start_time']):
                    available = False
                    break
            
            # If the station is available, assign the task
            if available:
                # If the task exceeds the deadline, adjust the start time
                if end_time > deadline:
                    start_time = deadline - total_time
                    end_time = deadline
                    print(f"Adjusted - Station {station} - Product {product} - Start: {start_time}, End: {end_time}")
                
                # If task can be completed on this station within the deadline
                if end_time <= deadline:
                    parallel_tasks.append({
                        'station': station,
                        'start_time': start_time,
                        'end_time': end_time,
                        'total_time': total_time,
                        'reconfig_time': reconfig_time,
                        'production_time': production_time,
                        'product': product,
                        'quantity': quantity,
                        'deadline': deadline
                    })
                    
                    # Update the station's end time
                    station_end_times[station] = end_time
                    station_timelines[station].append({
                        'product': product,
                        'quantity': quantity,
                        'start_time': start_time,
                        'end_time': end_time,
                        'deadline': deadline
                    })
                    assigned = True
                    break
        
        # If no station could meet the deadline, attempt to shift the task to any available station
        if not assigned:
            for station in product_to_station[product]:
                last_task = station_timelines[station][-1] if station_timelines[station] else None
                
                reconfig_time, production_time, total_time = calculate_task_times(last_task, product, quantity, reconfig_times, prod_times)
                
                # Try to start task at the earliest available time on this station
                start_time = max(station_end_times[station], last_task['end_time'] if last_task else start_date)
                end_time = start_time + total_time
                
                # Debug: Print task start and end times
                print(f"Shifting task - Station {station} - Product {product} - Start: {start_time}, End: {end_time}")
                
                # Check if the station is available at the desired time (no overlap)
                available = True
                for task in station_timelines[station]:
                    if (start_time < task['end_time']) and (end_time > task['start_time']):
                        available = False
                        break
                
                # If the station is available, assign the task
                if available:
                    # If the task exceeds the deadline, adjust the start time
                    if end_time > deadline:
                        start_time = deadline - total_time
                        end_time = deadline
                        print(f"Adjusted Shift - Station {station} - Product {product} - Start: {start_time}, End: {end_time}")
                    
                    parallel_tasks.append({
                        'station': station,
                        'start_time': start_time,
                        'end_time': end_time,
                        'total_time': total_time,
                        'reconfig_time': reconfig_time,
                        'production_time': production_time,
                        'product': product,
                        'quantity': quantity,
                        'deadline': deadline
                    })
                    
                    # Update the station's end time
                    station_end_times[station] = end_time
                    station_timelines[station].append({
                        'product': product,
                        'quantity': quantity,
                        'start_time': start_time,
                        'end_time': end_time,
                        'deadline': deadline
                    })
                    break

    return parallel_tasks


def update_station_timelines(parallel_tasks, station_timelines, total_times):
    """Updates station timelines and total times."""
    for task in parallel_tasks:
        station = task['station']
        station_timelines[station].append({
            'product': task['product'],
            'quantity': task['quantity'],
            'start_time': task['start_time'],
            'end_time': task['end_time'],
            'deadline': task['deadline']
        })
        total_times[station]['production'] += task['production_time'].total_seconds() / 60
        total_times[station]['reconfig'] += task['reconfig_time'].total_seconds() / 60

def optimize_schedule(products, product_to_station, reconfig_times, prod_times, demands, start_date):
    """Optimizes the schedule and returns station timelines and summary metrics."""
    
    station_timelines = {station: [] for stations in product_to_station.values() for station in stations}
    
    demands.sort(key=lambda x: x['deadline'])  # Sort tasks by deadline
    
    total_times = {station: {'production': 0, 'reconfig': 0} for station in station_timelines}

    parallel_tasks = allocate_tasks_to_stations(demands, product_to_station, station_timelines, reconfig_times, prod_times, start_date)
    
    update_station_timelines(parallel_tasks, station_timelines, total_times)
    
    return station_timelines, total_times

def generate_csv_outputs(station_timelines, total_times):
    """Writes the scheduling and summary results to CSV files."""
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

def generate_gantt_chart_matplotlib():
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

# Main Workflow
start_date = "20.01.2018 06:00:00"
start_date = datetime.strptime(start_date, '%d.%m.%Y %H:%M:%S')
products, product_to_station, reconfig_times, prod_times, demands = parse_inputs()
station_timelines, total_times = optimize_schedule(products, product_to_station, reconfig_times, prod_times, demands, start_date)
generate_csv_outputs(station_timelines, total_times)
generate_gantt_chart_matplotlib()
