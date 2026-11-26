# Physical internal stages on y' = λy:
#   (1) max |R_j| on the real stability interval [-β, 0]
#   (2) stage times c_j = R_j'(0): whether they increase monotonically (no backward steps)
# CRKG3 / CRKG4 should pass both. DUMKA3 fails |R_j|. ROCK4 / DUMKA3 may go backward in c_j.
#
#   julia --project=. tests/test_physical.jl

include(joinpath(@__DIR__, "..", "methods", "CRKG3.jl"))
include(joinpath(@__DIR__, "..", "methods", "CRKG4.jl"))

using OrdinaryDiffEqStabilizedRK, Printf
const SRK = OrdinaryDiffEqStabilizedRK

###############################################################################
# ROCK4 stages (Abdulle 2002 tables)

const ROCK4C = SRK.ROCK4ConstantCache(Float64, Float64, [0.0])
recf_start(k) = 1 + sum((2ROCK4C.ms[i] - 1 for i in 1:(k - 1)); init = 0)

function rock4_stages(k, x)
    m, st = ROCK4C.ms[k], recf_start(k)
    Ym, Y = one(x), one(x) + ROCK4C.recf[st] * x
    S = [Y]
    for i in 2:m
        μ, κ = ROCK4C.recf[st + 2(i - 2) + 1], ROCK4C.recf[st + 2(i - 2) + 2]
        Ym, Y = Y, μ * x * Y + (1 + κ) * Y - κ * Ym
        push!(S, Y)
    end
    a, b = ROCK4C.fpa[k], ROCK4C.fpb[k]
    k1 = x * Y
    F2 = Y + a[1] * k1
    k2 = x * F2
    F3 = Y + a[2] * k1 + a[3] * k2
    k3 = x * F3
    F4 = Y + a[4] * k1 + a[5] * k2 + a[6] * k3
    k4 = x * F4
    R = Y + b[1] * k1 + b[2] * k2 + b[3] * k3 + b[4] * k4
    return vcat(S, [F2, F3, F4]), R
end

function rock4_beta(k; n = 50_000)
    s = ROCK4C.ms[k] + 4
    xs = range(0.0, -0.6 * s^2; length = n)
    i = findfirst(x -> abs(rock4_stages(k, x)[2]) > 1 + 1e-12, xs)
    return i === nothing ? -xs[end] : -xs[i - 1]
end

###############################################################################
# DUMKA3 stages (Medovikov 1998 tables)

const DUMKA3C = SRK.DUMKA3ConstantCache(Float64, Float64, [0.0])

function dumka3_stages(k, x)
    r₁, r₂ = SRK.DUMKA3_first[k], SRK.DUMKA3_last[k]
    u = one(x)
    z0 = x * u          # f(u) with dt = 1, x = hλ
    S = typeof(u)[]
    for row in r₁:r₂
        c = DUMKA3C.coef[row]
        a₂₁, a₃₁, a₃₂, a₄₃ = c[1], c[2], c[3], c[6]
        Y = u + a₂₁ * z0
        push!(S, Y)
        z1 = x * Y
        Y = u + a₃₁ * z0 + a₃₂ * z1
        push!(S, Y)
        z0 = x * Y
        u = Y + a₄₃ * z0
        push!(S, u)
        z0 = x * u
    end
    return S, u
end

###############################################################################
# CRKG chain stages (physical by construction: R_j of a truncated chain)

function crkg_one_chain_stages(ch, L, x)
    S = Float64[]
    Y, Ym = 1.0, 0.0
    for j in 1:L
        F = x * Y
        push!(S, Y)
        j == L && break
        if j == 1
            Ym, Y = Y, Y + ch.mt[1] * F
        else
            Yn = ch.mu[j] * Y + ch.ka[j] * Ym + ch.mt[j] * F
            Ym, Y = Y, Yn
        end
    end
    return S
end

function crkg_chain_stages(m, x)
    # all stages of every chain (concatenated), for |R_j| scan
    reduce(vcat, (crkg_one_chain_stages(ch, m.L, x) for ch in m.chains))
end

###############################################################################
# scan max |R_j| on [-β, 0]

function max_abs_stage(stages_at, β; n = 20_000)
    best, jbest, xbest = 0.0, 0, 0.0
    for x in range(-β, 0.0; length = n)
        S = stages_at(x)
        v, j = findmax(abs, S)
        v > best && ((best, jbest, xbest) = (v, j, x))
    end
    return best, jbest, xbest
end

# stage times c_j = R_j'(0) by central difference; report whether c is strictly increasing
function stage_times(stages_at; δ = 1e-7)
    Sp, Sm = stages_at(δ), stages_at(-δ)
    return (Sp .- Sm) ./ (2δ)
end

function mono_report(c)
    nback = count(i -> c[i] <= c[i - 1], 2:length(c))
    cmin, cmax = minimum(c), maximum(c)
    return nback, cmin, cmax, nback == 0
end

const TOL = 1 + 1e-8
betas_check = (50, 200, 500)

@printf("%-8s %6s %8s %8s %12s %10s\n", "method", "β", "L/s", "stage", "max|R_j|", "physical?")
println("-"^60)

# CRKG3 / CRKG4
for (name, mod) in (("CRKG3", CRKG3), ("CRKG4", CRKG4))
    for β in betas_check
        M = mod.min_M(β; nu = 0.5, damp = mod.DAMP)
        m = mod.design(M; nu = 0.5, damp = mod.DAMP)
        βm = min(β, m.beta)
        mx, j, x = max_abs_stage(z -> crkg_chain_stages(m, z), βm)
        ok = mx <= TOL
        @printf("%-8s %6.0f %8d %8d %12.4e %10s\n", name, βm, m.L, j, mx, ok ? "yes" : "no")
    end
end

# ROCK4: pick tabulated degree whose β covers the target
for β in betas_check
    k = findfirst(i -> rock4_beta(i) >= β, eachindex(ROCK4C.ms))
    k === nothing && (k = length(ROCK4C.ms))
    βk = rock4_beta(k)
    s = ROCK4C.ms[k] + 4
    mx, j, x = max_abs_stage(z -> rock4_stages(k, z)[1], min(β, βk))
    ok = mx <= TOL
    @printf("%-8s %6.0f %8d %8d %12.4e %10s\n", "ROCK4", min(β, βk), s, j, mx, ok ? "yes" : "no")
end

# DUMKA3: pick tabulated degree whose stab_reg covers β
for β in betas_check
    k = findfirst(i -> SRK.DUMKA3_stab_reg[i] >= β, eachindex(SRK.DUMKA3_degrees))
    k === nothing && (k = length(SRK.DUMKA3_degrees))
    βk = SRK.DUMKA3_stab_reg[k]
    s = SRK.DUMKA3_degrees[k]
    mx, j, x = max_abs_stage(z -> dumka3_stages(k, z)[1], min(β, βk))
    ok = mx <= TOL
    @printf("%-8s %6.0f %8d %8d %12.4e %10s\n", "DUMKA3", min(β, βk), s, j, mx, ok ? "yes" : "no")
end

###############################################################################
# stage-time monotonicity (no backward steps)

println()
@printf("%-8s %6s %8s %8s %10s %10s %12s\n",
        "method", "β", "L/s", "n_back", "c_min", "c_max", "mono incr?")
println("-"^70)

for (name, mod) in (("CRKG3", CRKG3), ("CRKG4", CRKG4))
    for β in betas_check
        M = mod.min_M(β; nu = 0.5, damp = mod.DAMP)
        m = mod.design(M; nu = 0.5, damp = mod.DAMP)
        # each chain is its own time sequence from y_n; all must be increasing
        nback_tot, cmin, cmax = 0, Inf, -Inf
        for ch in m.chains
            c = stage_times(z -> crkg_one_chain_stages(ch, m.L, z))
            nb, cmn, cmx, _ = mono_report(c)
            nback_tot += nb
            cmin = min(cmin, cmn)
            cmax = max(cmax, cmx)
        end
        ok = nback_tot == 0
        @printf("%-8s %6.0f %8d %8d %10.4f %10.4f %12s\n",
                name, min(β, m.beta), m.L, nback_tot, cmin, cmax, ok ? "yes" : "no")
    end
end

for β in betas_check
    k = findfirst(i -> rock4_beta(i) >= β, eachindex(ROCK4C.ms))
    k === nothing && (k = length(ROCK4C.ms))
    βk = rock4_beta(k)
    s = ROCK4C.ms[k] + 4
    c = stage_times(z -> rock4_stages(k, z)[1])
    nback, cmin, cmax, ok = mono_report(c)
    @printf("%-8s %6.0f %8d %8d %10.4f %10.4f %12s\n",
            "ROCK4", min(β, βk), s, nback, cmin, cmax, ok ? "yes" : "no")
end

for β in betas_check
    k = findfirst(i -> SRK.DUMKA3_stab_reg[i] >= β, eachindex(SRK.DUMKA3_degrees))
    k === nothing && (k = length(SRK.DUMKA3_degrees))
    βk = SRK.DUMKA3_stab_reg[k]
    s = SRK.DUMKA3_degrees[k]
    c = stage_times(z -> dumka3_stages(k, z)[1])
    nback, cmin, cmax, ok = mono_report(c)
    @printf("%-8s %6.0f %8d %8d %10.4f %10.4f %12s\n",
            "DUMKA3", min(β, βk), s, nback, cmin, cmax, ok ? "yes" : "no")
end
