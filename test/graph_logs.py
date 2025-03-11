import csv
import matplotlib.pyplot as plt
import sys
import argparse

def plot_computation_times(csv_file, output_file, title=None, show_averages=False):
    # Read CSV file with no headers using csv library
    local_times = []
    distributed_times = []
    sizes = []

    
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) < 6:
                print(f"Error: Row does not have enough columns. Found {len(row)} columns, need at least 6.")
                continue
            
            try:
                local_times.append(float(row[4]))
                distributed_times.append(float(row[5]))
                sizes.append((int(row[0]), int(row[1]), int(row[2]), int(row[3])))
            except ValueError:
                print(f"Warning: Could not convert values to float in row: {row}")
                continue
    
    if not local_times or not distributed_times:
        print("Error: No valid data found in the CSV file.")
        sys.exit(1)
    
    # Calculate averages
    local_avg = sum(local_times) / len(local_times)
    distributed_avg = sum(distributed_times) / len(distributed_times)
    
    # Determine if either series needs to be in seconds
    use_seconds = max(max(local_times), max(distributed_times)) >= 1000
    
    # Convert all times to seconds if any time is >= 1000ms
    display_local_times = local_times
    display_distributed_times = distributed_times
    
    if use_seconds:
        display_local_times = [t / 1000.0 for t in local_times]
        display_distributed_times = [t / 1000.0 for t in distributed_times]
        unit_label = "seconds"
        local_avg_display = local_avg / 1000.0
        distributed_avg_display = distributed_avg / 1000.0
    else:
        unit_label = "milliseconds"
        local_avg_display = local_avg
        distributed_avg_display = distributed_avg
    
    # Create test case numbers for x-axis
    test_cases = range(1, len(local_times) + 1)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Bar width
    width = 0.35
    
    # Create bars with appropriate labels
    if show_averages:
        local_label = f'Local Hardware (Avg: {local_avg_display:.2f})'
        distributed_label = f'Distributed Hardware (Avg: {distributed_avg_display:.2f})'
    else:
        local_label = 'Local Hardware'
        distributed_label = 'Distributed Hardware'
    
    plt.bar([x - width/2 for x in test_cases], display_local_times, width, label=local_label)
    plt.bar([x + width/2 for x in test_cases], display_distributed_times, width, label=distributed_label)
    
    # Adding labels and title
    plt.xlabel('Test Case')
    plt.ylabel(f'Computation Time ({unit_label})')
    
    # Use custom title if provided, otherwise use default
    if title:
        plt.title(title)
    else:
        plt.title('Comparison of Computation Times: Local vs Distributed Hardware')
    
    if "diagnostic" in title.lower():
        plt.xticks(test_cases, [f"{ar}x{ac}@{br}x{bc}" for ar, ac, br, bc in sizes])
    else:
        plt.xticks(test_cases)
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot computation times from CSV data.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('output_file', help='Path to save the output plot')
    parser.add_argument('--title', '-t', help='Custom plot title')
    parser.add_argument('--averages', '-a', action='store_true', help='Show averages in legend')
    
    args = parser.parse_args()
    
    plot_computation_times(args.csv_file, args.output_file, args.title, args.averages)