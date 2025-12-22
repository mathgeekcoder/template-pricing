# Template Pricing

This repository contains code used in the journal publication:

"Accelerating Column Generation in Highly Degenerate Integer Programming Problems with Template Pricing"

If you use this code in your research, please cite the paper above.

## Features

- A column generation framework and several pricers.
- Uses HiGHS and optionally Gurobi (when available).
- Taskflow-based parallelization and lightweight logging integration.

## Requirements

- A C++20-capable compiler
- CMake >= 3.15
- vcpkg (recommended) to install third-party dependencies
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
# example, adjust path/config as needed
./colgen_pricing --help
```

## Suggested citation

If a formal BibTeX entry is not yet available, you can use the following placeholder and replace the authors, year and journal/volume/page with the actual values:

```bibtex
@article{template_pricing:XXXX,
  title = {Accelerating Column Generation in Highly Degenerate Integer Programming Problems with Template Pricing},
  author = {Author, A. and Author, B.},
  journal = {Journal Name},
  year = {XXXX},
  note = {Code: https://github.com/mathgeekcoder/template-pricing}
}
```

## License

This repository is provided under the terms of the included `LICENSE` file.

## Contact / Questions

If you find bugs, need help building, or want to contribute, please open an issue or pull request on the repository.
