# CRKG — Chain Runge–Kutta–Gegenbauer methods

Stabilized explicit Runge–Kutta schemes of orders 3 and 4 with physical, monotone-in-time stages and low internal amplification, intended for stagewise limiting on stiff parabolic and mixed problems, with a focus on viscous CFD. 

## Layout

- `methods/` — CRKG3 / CRKG4 implementations and precomputed tableaus (`crkg*_cache/default/`)
- `elixirs/` — demos of the paper experiments, including the CFD problems
- `tests/` — convergence, physical stages, and internal amplification checks
- `data/` — numerical outputs and figures used in the paper (see below)

## Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Tests

```bash
julia --project=. tests/test_vanderpol.jl
julia --project=. tests/test_physical.jl
julia --project=. tests/internal_stability_stabilized_rk.jl
```

## Elixirs

```bash
julia --project=. elixirs/elixir_advection_diffusion_1d.jl
julia --project=. -t 256 elixirs/elixir_navierstokes_daru_tenaud.jl
julia --project=. -t 256 elixirs/elixir_navierstokes_daru_tenaud_graded.jl
julia --project=. -t 256 elixirs/elixir_daru_timing_comparison.jl
julia --project=. elixirs/elixir_mp.jl          # GPU mixed precision (requires a CUDA device)
```

The Navier–Stokes demos are expensive, use more threads on a workstation or cluster as needed. The mixed-precision elixir needs an NVIDIA GPU and CUDA.jl.

## Data

- `data/AD/` — limited 1D advection–diffusion profiles shown in the paper (`Δt = 1/31` and `1/100`) and the composite figure
- `data/amplification/` — internal amplification JSON
- `data/VdP/` — van der Pol convergence JSON and figure
- `data/mp_gpu/` — mixed-precision GPU timing metrics
- `data/daru/` — Daru–Tenaud viscous shock-tube results (~90 MB total):
  - `daru_timing_Re200.tar.gz`, `daru_timing_Re200_P4.tar.gz` — final checkpoints and `metrics.json`
  - `daru_uniform_crkg3_Re200.tar.gz`, `daru_mild_graded_crkg3_Re200.tar.gz`, `daru_strongly_graded_crkg3_Re200.tar.gz` — final `t_0100.jld2` for the Re 200 schlieren / skin-friction cases
  - `daru_demos_schlieren.png`, `daru_skin_friction_Re200.csv`, `daru_skin_friction_Re200.png`
  - **Additional Daru results:** `daru_crkg3_Re500.tar.gz`, `daru_crkg4_Re1000.tar.gz` (final state; Re 1000 archive also includes the temperature figure)

Unpack with `tar -xzf <archive>.tar.gz` in `data/daru/`.
If full results, such as intermediate time solution data or grid data, is required, please reach out to the author, as these are very large files.

## Citation

If you use this code, please cite the accompanying paper (CRKG methods).
