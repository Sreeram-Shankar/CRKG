# CRKG — Chain Runge–Kutta–Gegenbauer methods

Stabilized explicit Runge–Kutta schemes of orders 3 and 4 with physical, monotone-in-time stages and low internal amplification, intended for stagewise limiting on stiff parabolic / compressible Navier–Stokes problems.

## Layout

- `methods/` — CRKG3 / CRKG4 implementations and precomputed tableaus (`crkg*_cache/default/`)
- `elixirs/` — demos of the paper experiments, including the CFD problems
- `tests/` — convergence, physical stages, and internal amplification checks

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

## Citation

If you use this code, please cite the accompanying paper (CRKG methods).
