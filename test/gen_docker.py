import os
import sys
import json
import yaml

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)

def generate_worker(id: int, network="csce689-project-network", image:str=None, min_mult=1024):
    worker = {'build': '.'}
    worker["build"] = "."
    if image:
        worker["image"] = f"{image}"
    worker["container_name"] = f"worker{id}"
    worker["depends_on"] = ["coordinator"]
    worker["command"] = f"python -u worker.py" #-m {min_mult}"
    worker["networks"] = [network]
    worker["environment"] = [f"NODE_ID={id}",
                             "COORDINATOR_HOST=coordinator",
                             "COORDINATOR_PORT=5000",
                             f"PORT={5000+id}",
                             f"MIN_MULT={min_mult}"]
    return worker

def generate_coordinator(network="csce689-project-network"):
    coordinator = {"build": ".",
                   "ports": ["5000:5000"],
                   "command": "python -u coordinator.py",
                   "volumes": ["./app/:/app"],
                   "networks": [network],
                   "environment": ["NODE_ID=coordinator",
                                   "PORT=5000"]}
    return coordinator

def generate_services(network="csce689-project-network", num_of_workers=7, images=list(), min_mult=16):
    services = dict()
    services["coordinator"] = generate_coordinator(network)

    for i in range(num_of_workers):
        image = None
        if i < len(images):
            image = images[i]
        services[f"worker{i+1}"] = generate_worker(i+1, network, image, min_mult)
    return services

def generate_network(name="csce689-project-network"):
    network = {name : {"driver" : "bridge"}}
    return network

def pull_config_from_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def write_compose_to_file(filepath, data):
    with open(filepath, 'w') as f:
        yaml.dump(data, f)


def main(default_file=None):
    if default_file:
        data = pull_config_from_file(default_file)
        num_of_workers = data['num_of_workers']
        images = data['images']
        min_mult = data['min_mult']

        docker = dict()
        docker["networks"] = generate_network() 
        docker["services"] = generate_services(num_of_workers=num_of_workers, images=images, min_mult=min_mult)
    else:
        print("Welcome to the Docker Compose Generator!")
        use_file = input("Would you like to read a configuration from a file? (y|n): ")
        if use_file.lower() == 'y':
            data = None
            while not data:
                try:
                    filepath = input("What is the file path of the configuration file? (from root project directory): ")
                    data = pull_config_from_file("../"+filepath)
                except Exception as e:
                    print(f"Error reading file: {e}")
                    print("Please try again")
            num_of_workers = data['num_of_workers']
            images = data['images']
            min_mult = data['min_mults']
        else:
            print()
            num_of_workers = int(input("How many workers would you like?: "))
            images = input("What specific images would you like to use? (comma-separated no-spaces list): ")
            if images:
                images = images.split(",")
            min_mult = int(input("What multiplication thresholds would you like?: "))


        docker = dict()
        docker["networks"] = generate_network() 
        docker["services"] = generate_services(num_of_workers=num_of_workers, images=images, min_mult=min_mult)
    write_compose_to_file("docker-compose.yml", docker)

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '-f':
        file = sys.argv[2]
    else:
        file=None
    main(default_file=file)
    print("Success!")