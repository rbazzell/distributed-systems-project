import os
import time
from rich.console import Console

console = Console()

def run(command: str, output=False):
    if not output:
        command += " > /dev/null 2>&1"
    return os.system(command)

def run_client(parameters:str, log_file: str, output=False):
    command = f"python app/client.py http://localhost:5000 {parameters} -l {log_file}"
    return run(command, output=output)

def complete(a):
    return a == 0

def error(a):
    return a != 0

def create_docker(generator_file):
    with console.status("[bold yellow]Initiating Docker setup...") as status:
        run(f"python test/gen_docker.py -f {generator_file}")
        console.print("Docker files generated!")
        run("docker compose up -d", output=True)
    console.print("[green]Docker initiated!")

def perform_tests(test_name, tests, log_file):
    failed_tests = False
    with console.status("[bold yellow]Running tests...") as status:
        for i, test in enumerate(tests):
            name, command, expectation = test
            result = run_client(command, "")
            if expectation(result):
                console.print(f"[green] Test {i + 1} - {name} succeeded!")
            else:
                console.print(f"[red] Test {i + 1} - {name} failed!")
                failed_tests = True
    if failed_tests:
        console.print(f"[red]Not all {test_name} tests succeeded!")
    else:
        console.print(f"[green]All {test_name} tests succeeded!\n\n")

def clear_docker():
    with console.status("[bold yellow]Closing existing Docker containers..."):
        run("docker compose kill")
        run("docker compose rm -f")




if __name__ == '__main__':
    options = {0 : "safe exit",
               1 : "small_scale (fastest for testing accuracy)",
               2 : "medium_scale",
               3 : "large_scale (only start to see performance gains here)"}

    while True:
        for num, option in options.items():
            console.print(f"({num}) : {option}")

        option = int(input("Which testing suite would you like to run? (press 0 to quit): "))

        clear_docker()
        match option:
            case 0:
                console.print("[blue]Goodbye!")
                exit()
            case 1:
                tests = [("1x1 @ 1x1", "1,1 1,1", complete), 
                         ("2x2 @ 2x2", "2,2 2,2", complete),
                         ("3x3 @ 3x3", "3,3 3,3", complete),
                         ("4x4 @ 4x4", "4,4 4,4", complete),
                         ("3x4 @ 4x5", "3,4 4,5", complete),
                         ("Mismatched sizes", "4,3 6,5", error),] 
                generator_file = "test/generators/small_scale.json"

                create_docker(generator_file)
                perform_tests("small scale", tests, "")
            case 2:
                tests = [("16x16 @ 16x16", "16,16 16,16", complete), 
                         ("17x13 @ 13x19", "17,13 13,19", complete),
                         ("47x1 @ 1x15", "30,1 1,15", complete),
                         ("31x31 @ 31x23", "31,31 31,23", complete),
                         ("32x32 @ 32x32", "32,32 32,32", complete),
                         ("64x64 @ 64x64", "64,64 64,64", complete),
                         ("128x128 @ 128x128", "128,128 128,128", complete),
                         ("256x256 @ 256x256", "256,256 256,256", complete),
                         ("Mismatched sizes", "16,17 30,31", error),]
                

                generator_file = "test/generators/medium_scale.json"

                create_docker(generator_file)
                perform_tests("medium scale", tests, "") 
            case 3:
                tests = [("16x16 @ 16x16", "16,16 16,16", complete), 
                         ("17x13 @ 13x19", "17,13 13,19", complete),
                         ("30x1 @ 1x15", "30,1 1,15", complete),
                         ("4x4 @ 4x4", "4,4 4,4", complete),
                         ("3x4 @ 4x5", "3,4 4,5", complete),
                         ("Mismatched sizes", "4,3 6,5", error),] 
                generator_file = "test/generators/large_scale.json"

                create_docker(generator_file)
                perform_tests("large scale", tests, "")
            case _:
                console.print("[bold red]Invalid option! Try again")