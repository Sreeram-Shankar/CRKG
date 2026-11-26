# Daru–Tenaud viscous shock tube on a uniform mesh
# with the CRKG methods

using Trixi
using LinearAlgebra, Printf
using Base.Threads

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "methods", "CRKG3.jl"))
using .CRKG3

method = CRKG3
Re = 200
base_level = 7
polydeg = 3
t_start, t_window = 0.0, 1.0
const C_h, C_p = 1.0, 1.0

mu() = 1.0 / Re
prandtl_number() = 0.73

function initial_condition_daru(x, t, equations)
    RealT = eltype(x)
    rho_rr = RealT(120)
    rho = x[1] > RealT(1 // 2) ? RealT(12 // 10) : rho_rr
    if abs(x[1] - RealT(1 // 2)) < RealT(1000) * eps(RealT)
        rho = RealT(1 // 2) * (RealT(12 // 10) + rho_rr)
    end
    p = rho / RealT(equations.gamma)
    return prim2cons(SVector(rho, zero(RealT), zero(RealT), p), equations)
end

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
surface_flux = flux_lax_friedrichs
mesh = TreeMesh((0.0, 0.0), (1.0, 1.0);
                initial_refinement_level = base_level, periodicity = (false, false))
wall = BoundaryConditionNavierStokesWall(NoSlip((x, t, eq) -> SVector(zero(eltype(x)), zero(eltype(x)))),
                                         Adiabatic((x, t, eq) -> zero(eltype(x))))
bch = (; x_neg = boundary_condition_slip_wall, x_pos = boundary_condition_slip_wall,
       y_neg = boundary_condition_slip_wall, y_pos = boundary_condition_slip_wall)
bcp = (; x_neg = wall, x_pos = wall, y_neg = wall, y_pos = wall)
basis = LobattoLegendreBasis(polydeg)
indicator = IndicatorEntropyCorrection(equations, basis)
volume_integral = VolumeIntegralAdaptive(indicator, VolumeIntegralWeakForm(),
                                         VolumeIntegralPureLGLFiniteVolume(surface_flux))
dg = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition_daru, dg;
                                             solver_parabolic = ParabolicFormulationLocalDG(),
                                             boundary_conditions = (bch, bcp))
ode = semidiscretize(semi, (t_start, t_start + t_window))
u0 = copy(ode.u0)
println(nameof(method), "  Re=", Re, "  level=", base_level,
        "  unknowns=", length(u0), "  elements=", Trixi.nelements(semi.solver, semi.cache),
        "  threads=", nthreads())

zs! = PositivityPreservingLimiterZhangShu(thresholds = (5e-6, 5e-6),
                                          variables = (Trixi.density, pressure))
lz! = PositivityPreservingLimiterLiuZhang(zs!, semi)
limiter!(u, t) = (lz!(u, nothing, semi, t); u)

const NRHS = Ref(0)
const PAR = similar(ode.u0)
function rhs!(du, u, t)
    NRHS[] += 1
    Trixi.rhs_hyperbolic!(du, u, semi, t)
    Trixi.rhs_parabolic!(PAR, u, semi, t)
    Trixi.@threaded for i in eachindex(du)
        @inbounds du[i] += PAR[i]
    end
    return du
end

const G2 = let ce = semi.cache.elements
    maximum(CartesianIndices(ce.inverse_jacobian)) do I
        i, j, e = Tuple(I)
        sum(d -> sum(abs2, Trixi.get_contravariant_vector(d, ce.contravariant_vectors, i, j, e)), 1:2) *
            ce.inverse_jacobian[i, j, e]^2
    end
end
const RHO = Ref((NaN, 0.0, 0.0))
const PART_RHO = fill(Inf, Threads.nthreads())
function radii(u, t = t_start)
    c = RHO[]
    c[1] == t && return c[2], c[3]
    dt_h = Trixi.calculate_dt(u, t_start, Returns(1.0), Returns(0.0), semi)
    ρh = polydeg / dt_h
    uw = Trixi.wrap_array(u, semi)
    nel = Trixi.nelements(semi.solver, semi.cache)
    nc = length(PART_RHO)
    Trixi.@threaded for c in 1:nc
        m = Inf
        for e in (div((c - 1) * nel, nc) + 1):div(c * nel, nc)
            for j in Trixi.eachnode(semi.solver), i in Trixi.eachnode(semi.solver)
                q = Trixi.get_node_vars(uw, equations, semi.solver, i, j, e)
                m = min(m, Float64(cons2prim(q, equations)[1]))
            end
        end
        PART_RHO[c] = m
    end
    ρmin = minimum(PART_RHO)
    ν_eff = Float64(equations_parabolic.mu) * Float64(equations_parabolic.max_visc_cond) / ρmin
    ρp = (5 / 4) * (polydeg + 1)^4 * ν_eff * G2 / 4
    s = 0.2 + 0.8 * min(max((t - t_start) / 1e-2, 0), 1)
    rh, rp = C_h * ρh / s, C_p * ρp / s
    RHO[] = (t, rh, rp)
    return rh, rp
end

u = copy(u0)
NRHS[] = 0
tt = @elapsed r = method.solve!(rhs!, u, (t_start, t_start + t_window);
                                limiter!, Mmax = 60, adaptive = :cfl_based,
                                rho_hyperbolic = (t, u) -> radii(u, t)[1],
                                rho_parabolic = (t, u) -> radii(u, t)[2],
                                safety_hyperbolic = 1.0, safety_parabolic = 1.0,
                                wave_fraction = 1 / sqrt(polydeg + 1))
@printf("steps=%d  M=%d  dt_last=%.4e  nf=%d  wall=%.2fs\n",
        r.accepted, r.M, r.dt_last, NRHS[], tt)
