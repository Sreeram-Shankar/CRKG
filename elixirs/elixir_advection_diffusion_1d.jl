# 1 dimensional advection–diffusion (ν=0.1) with Zhang–Shu / Liu–Zhang stagewise limiting.
# Demo: CRKG3 / DUMKA3 and CRKG4 / ROCK4 at the paper step sizes.
# You should see ROCK and DUMKA blowing up until smaller dts, while CRKG is nice

include(joinpath(@__DIR__, "..", "methods", "CRKG3.jl"))
include(joinpath(@__DIR__, "..", "methods", "CRKG4.jl"))
using Trixi, OrdinaryDiffEq, OrdinaryDiffEqStabilizedRK, Printf

t_final = 1.0
dts = (1 / 10, 1 / 31, 1 / 100, 1 / 500)
equations = LinearScalarAdvectionEquation1D(1.0)
equations_parabolic = LaplaceDiffusion1D(0.1, equations)
dg = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
mesh = TreeMesh(0.0, 1.0; initial_refinement_level = 4, periodicity = false)
left = (x, t, eq) -> SVector(one(eltype(x)))
right = (x, t, eq) -> SVector(1e-6)
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             (x, t, eq) -> SVector(1), dg;
                                             solver_parabolic = ParabolicFormulationLocalDG(),
                                             boundary_conditions = ((; x_neg = BoundaryConditionDirichlet(left),
                                                                       x_pos = boundary_condition_do_nothing),
                                                                    (; x_neg = BoundaryConditionDirichlet(left),
                                                                       x_pos = BoundaryConditionDirichlet(right))))
ode = semidiscretize(semi, (0.0, t_final))
cache_par, cache_hyp = similar(ode.u0), similar(ode.u0)
f_par, f_hyp = ode.f.f1, ode.f.f2
rhs! = (du, u, t) -> (f_par(cache_par, u, semi, t); f_hyp(cache_hyp, u, semi, t); @. du = cache_par + cache_hyp; du)
local_lim! = PositivityPreservingLimiterZhangShu(thresholds = (1e-6,), variables = (first,))
global_lim! = PositivityPreservingLimiterLiuZhang(local_lim!, semi; global_limiter_tol = 1e-6)
u_lim = copy(ode.u0)
limiter! = (u, t) -> (copyto!(vec(u_lim), u); global_lim!(u_lim, nothing, semi, t); u .= vec(u_lim); u)
stage_lim! = (u, integrator, P, t) -> (global_lim!(u, integrator, P, t); nothing)

u0 = Float64.(vec(ode.u0))
rho = CRKG4.estimate_rho(rhs!, u0, 0.0)

function run_crkg(mod, dt)
    y = copy(u0)
    mod.solve!(rhs!, y, (0.0, t_final), dt; limiter! = limiter!, adaptive = false)
    return y
end

function run_ode(Alg, dt)
    sol = solve(ode, Alg(eigen_est = i -> (i.eigen_est = rho));
                dt = dt, adaptive = false, stage_limiter = stage_lim!,
                save_everystep = false, dense = false)
    return Float64.(vec(sol.u[end]))
end

@printf("%-8s %10s %12s %12s %s\n", "method", "dt", "min u", "max u", "finite")
for dt in dts
    for (name, run) in (
        ("CRKG3", () -> run_crkg(CRKG3, dt)),
        ("DUMKA3", () -> run_ode(DUMKA3, dt)),
        ("CRKG4", () -> run_crkg(CRKG4, dt)),
        ("ROCK4", () -> run_ode(ROCK4, dt)),
    )
        u = run()
        @printf("%-8s %10.4g %12.4e %12.4e %s\n", name, dt, minimum(u), maximum(u), all(isfinite, u))
    end
end
