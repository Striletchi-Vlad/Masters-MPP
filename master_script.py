import os
import json


def main():
    prefix = "OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --oversubscribe --use-hwthread-cpus"
    # prefix = "mpirun "

    matrix_sizes = [100]
    # matrix_sizes = [1000, 10000]

    for ms in matrix_sizes:
        commands = [
            f"{prefix} -np 1 ./a.out --variant 1 --M {ms}",
            # f"{prefix} -np 1 ./a.out --variant 2a --M {ms} --threads 10",
            # f"{prefix} -np 1 ./a.out --variant 2a --M {ms} --threads 20",
            # f"{prefix} -np 1 ./a.out --variant 2a --M {ms} --threads 50",
            # f"{prefix} -np 1 ./a.out --variant 2b --M {ms} --threads 10",
            # f"{prefix} -np 1 ./a.out --variant 2b --M {ms} --threads 20",
            # f"{prefix} -np 1 ./a.out --variant 2b --M {ms} --threads 50",
            # f"{prefix} -np 1 ./a.out --variant 3a --M {ms} --threads 10",
            # f"{prefix} -np 1 ./a.out --variant 3a --M {ms} --threads 20",
            # f"{prefix} -np 1 ./a.out --variant 3a --M {ms} --threads 50",
            # f"{prefix} -np 1 ./a.out --variant 3b --M {ms} --threads 10",
            # f"{prefix} -np 1 ./a.out --variant 3b --M {ms} --threads 20",
            # f"{prefix} -np 1 ./a.out --variant 3b --M {ms} --threads 50",
            f"{prefix} -np 16 ./a.out --variant 4a --M {ms}",
            # f"{prefix} -np 36 ./a.out --variant 4a --M {ms}",
            # f"{prefix} -np 16 ./a.out --variant 4b --M {ms}",
            # f"{prefix} -np 36 ./a.out --variant 4b --M {ms}",
            # f"{prefix} -np 16 ./a.out --variant 5a --M {ms} --threads 10",
            # f"{prefix} -np 16 ./a.out --variant 5a --M {ms} --threads 20",
            # f"{prefix} -np 36 ./a.out --variant 5a --M {ms} --threads 10",
            # f"{prefix} -np 36 ./a.out --variant 5a --M {ms} --threads 20",
            # f"{prefix} -np 16 ./a.out --variant 5b --M {ms} --threads 10",
            # f"{prefix} -np 16 ./a.out --variant 5b --M {ms} --threads 20",
            # f"{prefix} -np 36 ./a.out --variant 5b --M {ms} --threads 10",
            # f"{prefix} -np 36 ./a.out --variant 5b --M {ms} --threads 20",
            # f"{prefix} -np 1 ./a.out --variant 6 --M {ms} --blockSize 1024",
            # f"{prefix} -np 1 ./a.out --variant 6 --M {ms} --blockSize 2048",
            # f"{prefix} -np 4 ./a.out --variant 7a --M {ms} --blockSize 1024",
            # f"{prefix} -np 4 ./a.out --variant 7a --M {ms} --blockSize 2048",
            # f"{prefix} -np 4 ./a.out --variant 7b --M {ms} --threads 10 --blockSize 1024",
            # f"{prefix} -np 4 ./a.out --variant 7b --M {ms} --threads 10 --blockSize 2048",
            # f"{prefix} -np 4 ./a.out --variant 7b --M {ms} --threads 20 --blockSize 1024",
            # f"{prefix} -np 4 ./a.out --variant 7b --M {ms} --threads 20 --blockSize 2048",
            # f"{prefix} -np 4 ./a.out --variant 7b --M {ms} --threads 50 --blockSize 1024",
            # f"{prefix} -np 4 ./a.out --variant 7b --M {ms} --threads 50 --blockSize 2048",
        ]

        for command in commands:
            print("Executing command: ", command)
            os.system(command)
            print("\n\n")

    print("All done! Building results table...")

    # Build results table
    results = {}
    # Scan local dir for jsons

    for filename in os.listdir("."):
        if filename.endswith(".json"):
            # Populate with speedup, which is computed as T_total improvement over T_total for variant 1 with same M
            if filename.startswith("1_"):
                results[filename] = json.load(open(filename))
                results[filename]["speedup"] = 1.0

    for filename in os.listdir("."):
        if filename.endswith(".json"):
            # Populate with speedup, which is computed as T_total improvement over T_total for variant 1 with same M
            if not filename.startswith("1_"):
                results[filename] = json.load(open(filename))
                m_part_of_string = filename.split("_")[1]
                original_filename = f"1_{m_part_of_string}_T1_B1.json"
                speedup = (
                    results[original_filename]["T_total"] / results[filename]["T_total"]
                )
                results[filename]["speedup"] = speedup

    # Print results
    # print("Results:")
    # for filename, result in results.items():
    #     print(f"File: {filename}")
    #     print(json.dumps(result, indent=4))
    #     print("\n")

    __import__("pprint").pprint(results)

    # Save to csv
    with open("results.csv", "w") as f:
        f.write("variant,T_reading,T_multiplication,T_writing,T_total,speedup\n")
        for filename, result in results.items():
            f.write(
                f"{filename},{result['T_reading']},{result['T_multiplication']},{result['T_writing']},{result['T_total']},{result['speedup']}\n"
            )
    print("Results saved to results.csv")


if __name__ == "__main__":
    main()
