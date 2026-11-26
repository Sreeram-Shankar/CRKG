# Internal amplification of CRKG3 / CRKG4 vs ROCK4 / DUMKA3
# (Ketcheson: for y'=λy, dt=1, inject ε at stage j and take θⱼ ≈ (y_pert − y_clean)/ε;
#  report max|θⱼ| on z∈[-β,0] with |ψ|≤1). Sweep β = 50:50:500.
#
#   julia --project=. tests/internal_stability_stabilized_rk.jl

include(joinpath(@__DIR__, "..", "methods", "CRKG3.jl"))
include(joinpath(@__DIR__, "..", "methods", "CRKG4.jl"))

using OrdinaryDiffEq, OrdinaryDiffEqStabilizedRK
using LinearAlgebra, Printf

const SRK = OrdinaryDiffEqStabilizedRK
const ROCK4C = SRK.ROCK4ConstantCache(Float64, Float64, [0.0])
const EPS = 1e-6
const NZ = 40
betas = 50:50:500

# ROCK4 real-interval length for tabulated degree index k (same rule as tests/test_physical.jl)
function rock4_beta(k; n = 50_000)
    m, st = ROCK4C.ms[k], 1 + sum((2ROCK4C.ms[i] - 1 for i in 1:(k - 1)); init = 0)
    function R(x)
        Ym, Y = one(x), one(x) + ROCK4C.recf[st] * x
        for i in 2:m
            μ, κ = ROCK4C.recf[st + 2(i - 2) + 1], ROCK4C.recf[st + 2(i - 2) + 2]
            Ym, Y = Y, μ * x * Y + (1 + κ) * Y - κ * Ym
        end
        a, b = ROCK4C.fpa[k], ROCK4C.fpb[k]
        k1 = x * Y
        k2 = x * (Y + a[1] * k1)
        k3 = x * (Y + a[2] * k1 + a[3] * k2)
        k4 = x * (Y + a[4] * k1 + a[5] * k2 + a[6] * k3)
        return Y + b[1] * k1 + b[2] * k2 + b[3] * k3 + b[4] * k4
    end
    s = m + 4
    xs = range(0.0, -0.6 * s^2; length = n)
    i = findfirst(x -> abs(R(x)) > 1 + 1e-12, xs)
    return i === nothing ? -xs[end] : -xs[i - 1]
end

# smallest tabulated stage count whose real stability interval covers β
function rock4_stages_for(β)
    k = findfirst(i -> rock4_beta(i) >= β, eachindex(ROCK4C.ms))
    k === nothing && (k = length(ROCK4C.ms))
    return ROCK4C.ms[k] + 4
end
function dumka3_stages_for(β)
    k = findfirst(i -> SRK.DUMKA3_stab_reg[i] >= β, eachindex(SRK.DUMKA3_degrees))
    k === nothing && (k = length(SRK.DUMKA3_degrees))
    return SRK.DUMKA3_degrees[k]
end

# one clean / perturbed step of an OrdinaryDiffEq method at a fixed stage count
function ode_step(alg, lam; j = nothing)
    n = Ref(0)
    lim!(u, _, _, _) = (n[] += 1; j == n[] && (u[1] += EPS); nothing)
    sol = solve(ODEProblem((du, u, p, t) -> (du[1] = p * u[1]), [1.0], (0.0, 1.0), lam),
                alg; dt = 1.0, adaptive = false, stage_limiter = lim!,
                save_everystep = false, dense = false,
                verbose = DEVerbosity(SciMLLogging.None()))
    return sol.u[end][1], n[]
end

# CRKG step with shared stages (matches the designed tableau)
function crkg_step(m, lam; j = nothing)
    L, k, chains, bs = m.L, m.share, m.chains, m.b
    y, s, n = 1.0, 0.0, 0
    Y, Ym, SF, Sk, Skm = y, 0.0, 0.0, 0.0, 0.0
    tr = chains[1]
    for jj in 1:L
        n += 1
        j == n && (Y += EPS)
        F = lam * Y
        jj == k + 1 && ((SF, Sk, Skm) = (F, Y, Ym))
        s += bs[1][jj] * F
        jj == L && break
        if jj == 1
            Ym, Y = Y, Y + tr.mt[1] * F
        else
            Yn = tr.mu[jj] * Y + tr.ka[jj] * Ym + tr.mt[jj] * F
            Ym, Y = Y, Yn
        end
    end
    for c in 2:length(chains)
        ch, b = chains[c], bs[c]
        if k > 0
            Y, Ym = Sk, Skm
        else
            Y = y
        end
        if k == 0
            Ym, Y = Y, Y + ch.mt[1] * SF
        else
            Yn = ch.mu[k + 1] * Y + ch.ka[k + 1] * Ym + ch.mt[k + 1] * SF
            Ym, Y = Y, Yn
        end
        for jj in (k + 2):L
            n += 1
            j == n && (Y += EPS)
            F = lam * Y
            s += b[jj] * F
            jj == L && break
            Yn = ch.mu[jj] * Y + ch.ka[jj] * Ym + ch.mt[jj] * F
            Ym, Y = Y, Yn
        end
    end
    return y + s, n
end

# max |θⱼ| over z∈[-β,0] with |ψ(z)|≤1
function max_amp(step; beta, nz = NZ)
    Mθ, z_at, j_at, nst = 0.0, 0.0, 0, 0
    for x in range(0.0, beta; length = nz)
        x == 0 && continue
        z = -x
        y0, n = step(z)
        (isfinite(y0) && abs(y0) <= 1 + 1e-5) || continue
        for jj in 1:n
            yj, _ = step(z; j = jj)
            th = abs((yj - y0) / EPS)
            isfinite(th) && th > Mθ && ((Mθ, z_at, j_at, nst) = (th, z, jj, n))
        end
    end
    return (; M = Mθ, z_at, j_at, s = nst)
end

results = Dict{String, Any}()

# ROCK4 / DUMKA3: same smallest covering stage count as tests/test_physical.jl
for (name, stages_for, make) in (
    ("ROCK4", rock4_stages_for, s -> ROCK4(min_stages = s, max_stages = s, eigen_est = i -> (i.eigen_est = 1.0))),
    ("DUMKA3", dumka3_stages_for, s -> DUMKA3(min_stages = s, max_stages = s, eigen_est = i -> (i.eigen_est = 1.0))),
)
    println(name)
    byβ = Dict{String, Any}()
    for β in betas
        s = stages_for(β)
        r = max_amp((lam; j = nothing) -> ode_step(make(s), lam; j); beta = β)
        byβ[string(Int(β))] = Dict("M" => r.M, "stages" => r.s, "z" => r.z_at, "stage" => r.j_at)
        @printf("  β=%4d  M=%.4e  stages=%d (target %d)\n", β, r.M, r.s, s)
    end
    results[name] = byβ
end

# CRKG: pick M so β(M) covers the reach, then scan that interval
for (name, mod) in (("CRKG3", CRKG3), ("CRKG4", CRKG4))
    println(name)
    byβ = Dict{String, Any}()
    for β in betas
        M = mod.min_M(β; nu = 0.5, damp = mod.DAMP)
        m = mod.design(M; nu = 0.5, damp = mod.DAMP)
        r = max_amp((lam; j = nothing) -> crkg_step(m, lam; j); beta = min(β, m.beta))
        byβ[string(Int(β))] = Dict("M_seg" => m.M, "L" => m.L, "beta" => m.beta, "share" => m.share,
                                   "M" => r.M, "stages" => r.s, "z" => r.z_at, "stage" => r.j_at)
        @printf("  β=%4d  M_seg=%d  L=%d  amp=%.4e  stages=%d\n", β, m.M, m.L, r.M, r.s)
    end
    results[name] = byβ
end
