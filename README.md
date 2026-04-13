# Template Pricing

This repository contains code used in the journal publication:

### [PLACEHOLDER UNTIL PUBLICATION]
```bibtex
@article{template_pricing:2026,
  title = {Accelerating Column Generation in Highly Degenerate Integer Programming Problems with Template Pricing},
  author = {Marshall, L. and Shah, P. and Dey, S.},
  journal = {Journal Name},
  year = {2026},
  note = {Code: https://github.com/mathgeekcoder/template-pricing}
}
```

## Features

- An optimized column generation implementation of GAP with several pricers (dantzig, pessoa, template).
- Lagrangian-relaxation implementation.
- Uses HiGHS and optionally Gurobi (when available).
- Taskflow-based parallelization and lightweight logging integration.

## Requirements

- C++20-capable compiler
- CMake >= 3.15
- vcpkg to install third-party dependencies
- Optional: Gurobi

## Minimal build instructions (vcpkg + CMake)

Below are minimal steps to build on linux using vcpkg.

1. Install compiler and build tools:

```bash
apt install cmake ninja-build curl zip unzip tar
```

2. Install and bootstrap vcpkg if you don't have it already. Follow the instructions at https://github.com/microsoft/vcpkg.

3. From the repository root:

```bash
# configure variables
export VCPKG_ROOT="$HOME/vcpkg"  # path to your vcpkg clone

# build
cmake --preset linux-release
cmake --build out/build/linux-release
```

Notes:
- If you installed dependencies system-wide or via another package manager, you can omit the vcpkg toolchain option, but CMake must be able to find the required packages.
- CMake will attempt to detect Gurobi automatically. To enable Gurobi support, set the environment variable `GUROBI_DIR` or `GUROBI_HOME` to your Gurobi installation directory before running CMake.

## Running

After a successful build the executable `colgen_pricing` will be produced in the build output directory. Run it with `--help` for available options:

```bash
./colgen_pricing --help
```

### Command-line arguments

Usage:

```bash
./colgen_pricing input [options]
```

`input` is required and can be an input file or directory (wildcard patterns `*` are supported).

#### Core options

- `-f, --force`  
  Force overwrite of existing log files.

- `--farkas_only`  
  Only perform RMP initialization.

- `-g, --gurobi`  
  Use Gurobi solver (only if built with Gurobi support).

- `--set_partition`  
  Use set-partition RMP instead of set-cover.

- `-m, --method METHOD`  
  Pricer method. Choices: `mt`, `lt`, `p`, `d`, `mip`, `lr`.  
  Default: `lt`.

- `-i, --init METHOD`  
  Farkas method. Choices: `mt`, `lt`, `d`, `auto`.  
  Default: `auto`.

- `-s, --seed N`  
  Random seed (`-1` use replication count for repeatibility).  
  Default: `-1`.

- `-r, --replications N`  
  Number of replications. Must be `>= 1`.  
  Default: `1`.

- `-p, --parallel N`  
  Number of parallel instances. Must be `>= 1`.  
  Default: `1`.

- `-t, --threads N`  
  Number of threads per parallel instance. Must be `>= 1`.  
  Default: hardware concurrency.

- `--time_limit SECS`  
  Time limit in seconds for each instance.  
  Default: `21600`.

#### Column management options

- `--func A B C`  
  Quadratic coefficients for column-age policy, interpreted as `ax^2 + bx + c` where `x = jobs/machines`.

- `--age_sweep START END STEP`  
  Sweep values for `age_limit`.

- `--multiplier_sweep START END STEP`  
  Sweep values for `max_col_multiplier`.

Notes for sweep mode:

- `--age_sweep` and `--multiplier_sweep` must be provided together.
- `--age_sweep` validation in code requires: `START >= 1`, `END >= START`, `STEP >= 1`.

#### Examples

```bash
# Basic run with defaults for method/init
./colgen_pricing data/instance.json

# Use lagrangian-relaxation, 4 threads
./colgen_pricing data -m lr -t 4 

# Provide custom quadratic retention function
./colgen_pricing data/instance.json -m lt --func 0 0 10

# Parameter sweep (both sweep flags required)
./colgen_pricing data/ --age_sweep 1 10 1 --multiplier_sweep 1 3 0.5
```

## Experiments

To reproduce the experiments in the paper, run the following commands (adjusting paths as needed).  

Each command produces many CSV files, one for each instance and replication. 
The `merge.py` script is then used to combine the individual CSV files into a single summary file for analysis. The `-x` flag to `merge.py` ignores all iterations except initialization and termination results.

### Statistics from results

To reproduce the statistics in the paper, run the following commands. These scripts reference the files in the `results` directory.

```bash
python src/gap/statistics_yagiura.py
python src/gap/statistics_isa.py
```


### Yagiura

#### Initialization only

```bash
./colgen_pricing instances/ISA/original -m d  --farkas_only --time_limit 600 -r 5
./colgen_pricing instances/ISA/original -m lt --farkas_only --time_limit 600 -r 5
./colgen_pricing instances/ISA/original -m mt --farkas_only --time_limit 600 -r 5
python merge.py *.csv -x -o yagiura-init-setcover.csv
```

```bash
./colgen_pricing instances/ISA/original -m d  --farkas_only --set_partition --time_limit 600 -r 5
./colgen_pricing instances/ISA/original -m lt --farkas_only --set_partition --time_limit 600 -r 5
./colgen_pricing instances/ISA/original -m mt --farkas_only --set_partition --time_limit 600 -r 5
python merge.py *.csv -x -o yagiura-init-setpartition.csv
```

#### Full column generation

```bash
./colgen_pricing instances/ISA/original -m d  --time_limit 21600 -r 5
./colgen_pricing instances/ISA/original -m p  --time_limit 21600 -r 5
./colgen_pricing instances/ISA/original -m lt --time_limit 21600 -r 5
./colgen_pricing instances/ISA/original -m mt --time_limit 21600 -r 5
python merge.py *.csv -x -o yagiura-results.csv
```

### ISA

#### Initialization only

```bash
./colgen_pricing instances/ISA/new_types -m d  --farkas_only --time_limit 600
./colgen_pricing instances/ISA/new_types -m lt --farkas_only --time_limit 600
./colgen_pricing instances/ISA/new_types -m mt --farkas_only --time_limit 600
./colgen_pricing instances/ISA/extended  -m d  --farkas_only --time_limit 600
./colgen_pricing instances/ISA/extended  -m lt --farkas_only --time_limit 600
./colgen_pricing instances/ISA/extended  -m mt --farkas_only --time_limit 600
python merge.py *.csv -x -o isa-init-setcover.csv
```

#### Full column generation

```bash
./colgen_pricing instances/ISA/new_types -m p  --time_limit 21600
./colgen_pricing instances/ISA/new_types -m lt --time_limit 21600
./colgen_pricing instances/ISA/new_types -m lr --time_limit 21600
./colgen_pricing instances/ISA/extended  -m p  --time_limit 21600
./colgen_pricing instances/ISA/extended  -m lt --time_limit 21600
./colgen_pricing instances/ISA/extended  -m lr --time_limit 21600
python merge.py *.csv -x -o isa-results.csv
```

## License

This repository is provided under the terms of the included `LICENSE` file.

## Contact / Questions

If you find bugs, need help building, or want to contribute, please open an issue or pull request on the repository.
