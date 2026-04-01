# Mesh_To_EllipsoidSDF

## Installation

Create the conda environment:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate ellipsoid-fit
```

## Usage

Interactive fitting tool:
```bash
python main.py
```

SDF method comparison:
```bash
python benchmark_sdf.py
```

Systematic SDF benchmark across varying aspect ratios:
```bash
python systematic_benchmark.py --plot
```

## Project Structure
The core optimization logic is implemented with the NVIDIA Warp framework and located in `optimization.py`.
The implementation of different ellipsoid SDF approcimations can be found in `sdf_methods.py`.
