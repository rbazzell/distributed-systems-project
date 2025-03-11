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

def create_docker(generator_file, output=False):
    with console.status("[bold yellow]Initiating Docker setup...") as status:
        run(f"python test/gen_docker.py -f {generator_file}")
        console.print("Docker files generated!")
        run("docker-compose up -d --build", output=output)
    console.print("[green]Docker initiated!")

def perform_tests(test_name, tests, log_file):
    failed_tests = False
    with console.status("[bold yellow]Running tests...") as status:
        for i, test in enumerate(tests):
            name, command, expectation = test
            result = run_client(command, log_file)
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
        run("docker-compose kill")
        run("docker-compose rm -f")

def clear_log_file(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)





if __name__ == '__main__':
    console.print("Diagnostic options perform tests on a variety of matrix sizes for that scale")
    console.print("Performance options perform tests on the same matrix size repeatedly")
    options = {0 : "Safe exit (closes Docker containers on the way out)",
               1 : "Small scale system (diagnostic)",
               2 : "Small scale system (performance)",
               3 : "Medium scale system (diagnostic)",
               4 : "Medium scale system (performance)",
               5 : "Large scale system (diagnostic)",
               6 : "Large scale system (performance)",
               7 : "Very large scale system (diagnostic)",
               8 : "Very large scale system (performance)"}

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
                         ("3x4 @ 4x8", "3,4 4,8", complete),] 
                generator_file = "test/generators/small_scale.json"
                log_file = "test/logs/small_scale_diagnostic.csv"

                create_docker(generator_file)
                clear_log_file(log_file)
                time.sleep(1)
                perform_tests("small scale diagnostic", tests, log_file)
            case 2:
                test = ("16x16 @ 16x16", "16,16 16,16", complete)
                tests = [test] * 20
                generator_file = "test/generators/small_scale.json"
                log_file = "test/logs/small_scale_performance.csv"

                create_docker(generator_file)
                clear_log_file(log_file)
                time.sleep(1)
                perform_tests("small scale performance", tests, log_file)
            case 3:
                tests = [("16x16 @ 16x16", "16,16 16,16", complete), 
                         ("17x13 @ 13x19", "17,13 13,19", complete),
                         ("47x1 @ 1x15", "30,1 1,15", complete),
                         ("31x31 @ 31x23", "31,31 31,23", complete),
                         ("32x32 @ 32x32", "32,32 32,32", complete),
                         ("64x64 @ 64x64", "64,64 64,64", complete),
                         ("128x128 @ 128x128", "128,128 128,128", complete),
                         ("256x256 @ 256x256", "256,256 256,256", complete),]
                

                generator_file = "test/generators/medium_scale.json"
                log_file = "test/logs/medium_scale_diagnostic.csv"

                create_docker(generator_file)
                clear_log_file(log_file)
                time.sleep(1)
                perform_tests("medium scale diagnostic", tests, log_file) 
            case 4:
                test = ("128x128 @ 128x128", "128,128 128,128", complete)
                tests = [test] * 20
                generator_file = "test/generators/medium_scale.json"
                log_file = "test/logs/medium_scale_performance.csv"

                create_docker(generator_file)
                clear_log_file(log_file)
                time.sleep(1)
                perform_tests("medium scale performance", tests, log_file)
            case 5:
                tests = [("256x256 @ 256x256", "256,256 256,256", complete), 
                         ("512x512 @ 512x512", "512,512 512,512", complete),
                         ("1024x1024 @ 1024x1024", "1024,1024 1024,1024", complete),
                         ("2048x2048 @ 2048x2048", "2048,2048 2048,2048", complete),
                         ("1024x32 @ 32x1024", "1024,64 64,1024", complete),] 
                generator_file = "test/generators/large_scale.json"
                log_file = "test/logs/large_scale_diagnostic.csv"

                create_docker(generator_file)
                clear_log_file(log_file)
                time.sleep(1)
                perform_tests("large scale diagnostic", tests, log_file)
            case 6:
                test = ("2048x2048 @ 2048x2048", "2048,2048 2048,2048", complete)
                tests = [test] * 20
                generator_file = "test/generators/large_scale.json"
                log_file = "test/logs/large_scale_performance.csv"

                create_docker(generator_file)
                clear_log_file(log_file)
                time.sleep(1)
                perform_tests("large scale performance", tests, log_file)
            case 7:
                tests = [("2048x2048 @ 2048x2048", "2048,2048 2048,2048", complete), 
                         ("4096x4096 @ 4096x4096", "4096,4096 4096,4096", complete),
                         ("8192x8192 @ 8192x8192", "8192,8192 8192,8192", complete),] 
                generator_file = "test/generators/very_large_scale.json"
                log_file = "test/logs/very_large_scale_diagnostic.csv"

                create_docker(generator_file)
                clear_log_file(log_file)
                time.sleep(1)
                perform_tests("very large scale diagnostic", tests, log_file)
            case 8:
                test = ("8192x8192 @ 8192x8192", "8192,8192 8192,8192", complete)
                tests = [test] * 20
                generator_file = "test/generators/very_large_scale.json"
                log_file = "test/logs/very_large_scale_performance.csv"

                create_docker(generator_file)
                clear_log_file(log_file)
                time.sleep(1)
                perform_tests("very large scale performance", tests, log_file)
            case _:
                console.print("[bold red]Invalid option! Try again")