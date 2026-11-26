#this file designs the "butcher tableau" for the method. it is the main file that designs the method and is used by the other files to get the method tableau.
# The process of designing the method is as follows:
# 1. Have the target stability polynomial, which is the one that we want to make the linaer combinations of different
# chains match the amplication region to. This is a Gegenbauer polynomial that has some additional damping stages (like RKC) 
#. The reason for the damping is becasue Gegenbauer polynomials can be quite oscillatory, and we want to dampen. In addition to
# that, we want the polynomial to be at the Gegenbaur parameter lambda = 0.5 by default, which recovers Legendre polynomials.
#. The reason for this is because this gives a good balance of damping and real axis stability extension. These polynomials
#  must also be of the correct order to the exponential for the stability function. the length is 3M for CRKG3 and 4M for CRKG4.
#
# 2. Have the Gegenbauer polynomials, which are like basis functions for the stability polynomial.
# For the 3rd order method, we want one more Gegenbauer polynomial (another chain), and for the 4th order method, we want two more.
# This allows linear combinations of the differnt chains to match the target stability polynomial while also satisfying the higher
# order nonlinear conditions. This is the chain trick which we have.
# The way we pick this is by taking a set of Gegenbauyar polynomials (various lambda values and damping stages and coverage in the -real axis), finding the ones with the lowest interanl amplicaition that satisfy what we want,
# then finding ones within 5x (by default) of the best one that also share some stages, which allows a significant cost reduction by sharing rhs evals.
# From here, there is one chain we must evaluate in full (the base chain, which we denote as the trunk), and then the other chains we can share stages with if 
# present. 

# 3. After we pick the Gegenbauer polynomials, we need to solve for the weights of the method. This is done by solving a system of equations that
# ensures that the stability polynomial matches the target stability polynomial, and that the nonlinear order conditions are satisfied.
# From there, we have built the method tableau for the given M, nu, damp, and amp_factor. by default, these are fixed to the aforementione values.
# you can change them if you want, but we do not recommend if you do change it, it will store the method in a subfokder of the method cache. The
# tableaus stored in the cache root are the default ones. 

#All the variables in CAPS are specific to each order and are defined in the module for each one

const DAMP = 0.5 # endpoint shift of the target; with the ETA damping level it only needs to make ETA attainable, so it is smaller than the classical RKC value
const ETA = 0.8  # damping level: |R| <= ETA on [-beta, -X0], which keeps the stability region a wide strip instead of pinching to zero width at interior extrema
const X0 = 0.5   # near the origin |R| ≈ e^x > ETA, so the damping level is only imposed for x <= -X0 (needs X0 > -log(ETA))

# Curated tableaus used at runtime are the default. saving a tableau in a subfolder of the cache root will make it the default for that order
cache_path(M) = joinpath(CACHE_DIR, "default", "M$M.jls")
variant_cache_dir(; nu = 0.5, damp = DAMP, amp_factor = 5.0) =
    joinpath(CACHE_DIR, "nu-$(@sprintf("%.6e", Float64(nu)))-damp-$(@sprintf("%.6e", Float64(damp)))-amp$(@sprintf("%.3g", Float64(amp_factor)))-eta-$(ETA)")
variant_cache_path(M; nu = 0.5, damp = DAMP, amp_factor = 5.0) =
    joinpath(variant_cache_dir(; nu, damp, amp_factor), "M$M.jls")

import HiGHS

###############################################################################
# Target stability polynomial. this is the one that we want to make the linaer combinations of different
# chains match the amplication region to. 

# A single Gegenbauer polynomial by its three-term recurrence, and its derivatives (which
# are also Gegenbauer polynomials with a shifted parameter, times a constant).
function gegC(lambda, n, w)
    n == 0 && return one(w)
    c0, c1 = one(w), 2lambda * w
    for k in 1:(n - 1)
        c0, c1 = c1, (2w * (k + lambda) * c1 - (k + 2lambda - 1) * c0) / (k + 1)
    end
    return c1
end
dC(lambda, n, j, w) =
    n < j ? zero(w) : prod(2(lambda + i) for i in 0:(j - 1); init = one(w)) * gegC(lambda + j, n - j, w)

# The polynomial's weights for a given interval length beta, making it match the
# exponential to order p at the origin. Each stage of each polynoial chain has its own weight,
# so the combined polynomial is a weighted sum of stages of all chains that has to satisfy the 
# order conditions.
function order_coeffs(nu, M, beta, w0; p = ORDER, dN = nothing)
    s = beta / (1 + w0)
    if p == ORDER
        d = [dC(nu, k * M, n, w0) for n in 1:ORDER, k in 1:ORDER] \ [s^n / 2 for n in 1:ORDER]
    else
        A = [dC(nu, k * M, n, w0) for n in 1:p, k in 1:p]
        d = [A \ [s^n / 2 - dN * dC(nu, ORDER * M, n, w0) for n in 1:p]; dN]
    end
    return 1 - 2 * sum(d[k] * gegC(nu, k * M, w0) for k in 1:ORDER), d
end
evalG(nu, M, d0, d, w) = d0 + 2 * sum(d[k] * gegC(nu, k * M, w) for k in 1:ORDER)

# The longest stable interval, coming from the algorithm of O'Sullivan modified for this code.
# The order conditions are imposed slightly past the endpoint to damp the interior.
# Away from the origin |R| is also kept below ETA, so that no interior extremum touches one.
function find_beta(nu, M; damp = DAMP, eta = ETA, x0 = X0)
    w0 = 1 + damp / M^2
    ok(beta) = begin
        d0, d = order_coeffs(nu, M, beta, w0)
        wc = w0 - (1 + w0) * x0 / beta # w at x = -x0
        all(range(-1, w0; length = 400M)) do w
            r = abs(evalG(nu, M, d0, d, w))
            r <= 1 + 1e-10 && (w > wc || r <= eta)
        end
    end
    betas = range(8M^2, 0.5; length = 800)
    i = findfirst(ok, betas)
    lo, hi = betas[i], betas[max(i - 1, 1)]
    for _ in 1:60
        mid = (lo + hi) / 2
        ok(mid) ? (lo = mid) : (hi = mid)
    end
    return lo
end

# The smallest M whose interval covers the step size times the spectral radius (for
# choosing the stage count of a fixed-step run). Floor is 2 — M = 1 is never useful.
function min_M(h_rho; nu = 0.5, damp = DAMP)
    M = max(2, floor(Int, sqrt(h_rho / 0.3) / ORDER) - 1)
    while find_beta(nu, M; damp) < h_rho
        M += 1
    end
    return M
end

###############################################################################
# Chains

# One chain: the recurrence coefficients for each stage, plus the real time at which each
# stage approximates the solution (as a fraction of the step).
struct Chain
    nu::Float64
    v0::Float64
    v1::Float64
    mu::Vector{Float64}
    ka::Vector{Float64}
    mt::Vector{Float64}
    c::Vector{Float64}
end

# Builds a chain whose stage polynomials are scaled to cover `coverage` times the target
# interval (so they stay bounded on it), with Gegenbauer parameter nu and damping eps.
# Coverage above 1 makes a chain amplify stage errors less. The stage times are the slopes of the stage
# polynomials at the origin (equivalently, the row sums of the Butcher matrix).
function chain(L, nu, eps, beta, coverage = 1.0)
    v0 = 1 + eps / L^2
    v1 = (1 + v0) / (coverage * beta)
    bj = [1 / gegC(nu, j, v0) for j in 0:L]
    mu, ka, mt = zeros(L), zeros(L), zeros(L)
    mt[1] = v1 / v0
    for j in 2:L
        mu[j] = 2(j - 1 + nu) / j * v0 * bj[j + 1] / bj[j]
        ka[j] = -(j + 2nu - 2) / j * bj[j + 1] / bj[j - 1]
        mt[j] = 2(j - 1 + nu) / j * v1 * bj[j + 1] / bj[j]
    end
    c = [j == 0 ? 0.0 : v1 * dC(nu, j, 1, v0) / gegC(nu, j, v0) for j in 0:(L - 1)]
    return Chain(nu, v0, v1, mu, ka, mt, c)
end

###############################################################################
# Order condition for nonlinear order. These are all stolen from established RK theory

# All rooted trees with at most p nodes. Each tree is one order condition; its "density"
# gamma gives the exact value the method must reproduce.
function trees(p)
    kids, order, gamma = [Int[]], [1], [1.0]
    for n in 2:p
        last = length(kids)
        function add!(ch, start, rem)
            if rem == 0
                push!(kids, copy(ch))
                push!(order, n)
                push!(gamma, n * prod((gamma[k] for k in ch); init = 1.0))
                return
            end
            for k in start:last
                if order[k] <= rem
                    push!(ch, k)
                    add!(ch, k, rem - order[k])
                    pop!(ch)
                end
            end
        end
        add!(Int[], 1, n - 1)
    end
    return kids, order, gamma
end

# "Tall" trees are already fixed by the stability polynomial.
istall(kids, i) = isempty(kids[i]) || (length(kids[i]) == 1 && istall(kids, kids[i][1]))

###############################################################################
# Weights

# Solves for the real stage weights. Constraints: the combined stability polynomial equals
# the target polynomial (checked at L Chebyshev points of the interval), and every
# non-tall order condition up to order p holds exactly. Among all solutions it balances
# small weights against a small (ORDER+1)-order error, then refines once to full precision.
# This comes from 0'Sullivan's paper to find the polynomials for linear order, 
# modifed with the extra damping terms we add. 
function solve_weights(chs, L, target, p, beta, kids, ord, gamma; lam = 1.0)
    xs = [-beta * (1 - cos((2i - 1) * pi / (2L))) / 2 for i in 1:L]
    stage(c, x) = [gegC(c.nu, j, c.v0 + c.v1 * x) / gegC(c.nu, j, c.v0) for j in 0:(L - 1)]
    stab = reduce(vcat, [transpose(reduce(vcat, [x .* stage(c, x) for c in chs])) for x in xs])
    # order-condition contributions per chain (Butcher A built from the recurrence)
    G = map(chs) do c
        A = zeros(L + 1, L)
        A[2, 1] = c.mt[1]
        for j in 2:L
            A[j + 1, :] = c.mu[j] .* A[j, :] .+ c.ka[j] .* A[j - 1, :]
            A[j + 1, j] += c.mt[j]
        end
        A = A[1:L, 1:L]
        G = Vector{Vector{Float64}}(undef, length(kids))
        AG = similar(G)
        for i in eachindex(kids)
            g = ones(L)
            for k in kids[i]
                g .*= AG[k]
            end
            G[i] = g
            AG[i] = A * g
        end
        G
    end
    row(t) = transpose(reduce(vcat, [G[c][t] for c in eachindex(chs)]))
    nt = findall(i -> !istall(kids, i) && ord[i] <= p, eachindex(kids))
    t_next = findall(==(ORDER + 1), ord)
    A = vcat(stab, isempty(nt) ? zeros(0, size(stab, 2)) : reduce(vcat, [row(t) for t in nt]))
    rhs = vcat([target(x) - 1 for x in xs], 1 ./ gamma[nt])
    E = reduce(vcat, [row(t) for t in t_next])
    e = 1 ./ gamma[t_next]
    F = svd(A)
    r = count(>(1e-13F.S[1]), F.S)
    x0 = F.V[:, 1:r] * ((F.U[:, 1:r]' * rhs) ./ F.S[1:r])
    Z = F.V[:, (r + 1):end]
    x = isempty(Z) ? x0 : x0 + Z * ([Z; sqrt(lam) .* (E * Z)] \ [-x0; sqrt(lam) .* (e .- E * x0)])
    x += F.V[:, 1:r] * ((F.U[:, 1:r]' * (rhs - A * x)) ./ F.S[1:r])
    res = norm(A * x - rhs) / (1 + norm(rhs))
    return [x[((c - 1) * L + 1):(c * L)] for c in eachindex(chs)], res, norm(E * x - e)
end

# The largest factor by which a change in any stage state reaches the new solution. internal amplicaiton minimzation.
# Samples the real interval [-beta, 0] and, if delta > 0, the imag segment [0, delta] used by the CFL box.
function output_amplification(chs, bs, L, beta; npts = 150, delta = 0.0)
    q = 0.0
    zs = ComplexF64[complex(x, 0.0) for x in range(-beta, 0; length = npts)]
    delta > 0 && append!(zs, ComplexF64[complex(0.0, y) for y in range(0, delta; length = npts)])
    for (c, b) in zip(chs, bs), z in zs, j in 0:(L - 1)
        Yp = zeros(ComplexF64, L + 1)
        Yp[j + 1] = 1
        resp = z * b[j + 1]
        for k in (j + 1):(L - 1)
            Yp[k + 1] = k == 1 ? (1 + c.mt[1] * z) * Yp[1] :
                        c.mu[k] * Yp[k] + c.ka[k] * Yp[k - 1] + c.mt[k] * z * Yp[k]
            resp += z * b[k + 1] * Yp[k + 1]
        end
        q = max(q, abs(resp))
    end
    return q
end

subsets(v, K) = K == 0 ? [eltype(v)[]] :
    [[v[i]; t] for i in eachindex(v) for t in subsets(v[(i + 1):end], K - 1)]

###############################################################################
# Shared stages. Chain 1 is the trunk. The other chains ("branches") reuse the trunk's
# first `share` stages and their right-hand sides, then continue with their own
# recurrence from the trunk's states. This is exact (no approximation): it is simply a
# different, cheaper set of stages, and the weights are solved for it in the same way.
# A design with share = k saves k right-hand sides per branch per step (and share = 0
# still shares f(u), saving one per branch).

# The method's stages as one list: the trunk's stages 0..L-1, then each branch's own
# stages share+1..L-1. Returns the list, a lookup (chain, index) -> position, the Butcher
# matrix of the whole list, and a function giving all stage polynomials at a point x.
function branched_system(chs, L, k)
    stages = Tuple{Int,Int}[]
    gid = Dict{Tuple{Int,Int},Int}()
    for j in 0:(L - 1)
        push!(stages, (1, j)); gid[(1, j)] = length(stages)
    end
    for c in 2:length(chs), j in (k + 1):(L - 1)
        push!(stages, (c, j)); gid[(c, j)] = length(stages)
    end
    pos(c, j) = (c == 1 || j <= k) ? gid[(1, j)] : gid[(c, j)]
    S = length(stages)
    A = zeros(S, S)
    for (s, (c, j)) in enumerate(stages)
        j == 0 && continue
        ch = chs[c]
        if j == 1
            A[s, :] = A[pos(c, 0), :]; A[s, pos(c, 0)] += ch.mt[1]
        else
            A[s, :] = ch.mu[j] .* A[pos(c, j - 1), :] .+ ch.ka[j] .* A[pos(c, j - 2), :]
            A[s, pos(c, j - 1)] += ch.mt[j]
        end
    end
    function polys(x)
        P = zeros(S)
        for (s, (c, j)) in enumerate(stages)
            ch = chs[c]
            P[s] = j == 0 ? 1.0 :
                   j == 1 ? P[pos(c, 0)] * (1 + ch.mt[1] * x) :
                   ch.mu[j] * P[pos(c, j - 1)] + ch.ka[j] * P[pos(c, j - 2)] + ch.mt[j] * x * P[pos(c, j - 1)]
        end
        return P
    end
    return stages, pos, A, polys
end

# Weights for a shared-stage design, solved exactly as in `solve_weights` (target
# stability polynomial at Chebyshev points, every order condition up to p, small
# (ORDER+1)-order error), but over the shared stage list.
function solve_weights_shared(chs, L, k, target, p, beta, kids, ord, gamma; lam = 1.0)
    stages, pos, A, polys = branched_system(chs, L, k)
    xs = [-beta * (1 - cos((2i - 1) * pi / (2L))) / 2 for i in 1:L]
    stab = reduce(vcat, [transpose(x .* polys(x)) for x in xs])
    G = Vector{Vector{Float64}}(undef, length(kids)); AG = similar(G)
    for i in eachindex(kids)
        g = ones(size(A, 1))
        for kk in kids[i]
            g .*= AG[kk]
        end
        G[i] = g; AG[i] = A * g
    end
    nt = findall(i -> !istall(kids, i) && ord[i] <= p, eachindex(kids)); t_next = findall(==(ORDER + 1), ord)
    Am = vcat(stab, isempty(nt) ? zeros(0, size(stab, 2)) : reduce(vcat, [transpose(G[t]) for t in nt]))
    rhs = vcat([target(x) - 1 for x in xs], 1 ./ gamma[nt])
    E = reduce(vcat, [transpose(G[t]) for t in t_next]); e = 1 ./ gamma[t_next]
    F = svd(Am)
    r = count(>(1e-13F.S[1]), F.S)
    x0 = F.V[:, 1:r] * ((F.U[:, 1:r]' * rhs) ./ F.S[1:r])
    Z = F.V[:, (r + 1):end]
    x = isempty(Z) ? x0 : x0 + Z * ([Z; sqrt(lam) .* (E * Z)] \ [-x0; sqrt(lam) .* (e .- E * x0)])
    x += F.V[:, 1:r] * ((F.U[:, 1:r]' * (rhs - Am * x)) ./ F.S[1:r])
    res = norm(Am * x - rhs) / (1 + norm(rhs))
    return x, res, norm(E * x - e), stages, pos, A
end

# The largest factor by which a change in any stage state reaches the new solution, for a
# shared-stage design (a change in a shared stage reaches every branch).
function output_amplification_shared(chs, L, k, w, beta; npts = 150)
    stages, pos, _, _ = branched_system(chs, L, k)
    S = length(stages)
    q = 0.0
    P = zeros(S)
    for x in range(-beta, 0; length = npts), s0 in 1:S
        fill!(P, 0.0); P[s0] = 1.0
        for (s, (c, j)) in enumerate(stages)
            (s <= s0 || j == 0) && continue
            ch = chs[c]
            P[s] = j == 1 ? P[pos(c, 0)] * (1 + ch.mt[1] * x) :
                   ch.mu[j] * P[pos(c, j - 1)] + ch.ka[j] * P[pos(c, j - 2)] + ch.mt[j] * x * P[pos(c, j - 1)]
        end
        q = max(q, abs(x * dot(w, P)))
    end
    return q
end

# Largest |stage| over all chains (trunk and branches) on the box [-beta, 0] × [-δ, δ], checked on its boundary
function stage_box_max(chs, L, k, beta, δ)
    function smax(z)
        T = [one(z), one(z) + chs[1].mt[1] * z]
        for j in 2:(L - 1)
            push!(T, chs[1].mu[j] * T[j] + chs[1].ka[j] * T[j - 1] + chs[1].mt[j] * z * T[j])
        end
        s = maximum(abs, T)
        for c in chs[2:end]
            Ym, Y = k > 0 ? (T[k + 1], c.mu[k + 1] * T[k + 1] + c.ka[k + 1] * T[k] + c.mt[k + 1] * z * T[k + 1]) :
                            (one(z), one(z) + c.mt[1] * z)
            s = max(s, abs(Y))
            for j in (k + 2):(L - 1)
                Ym, Y = Y, c.mu[j] * Y + c.ka[j] * Ym + c.mt[j] * z * Y
                s = max(s, abs(Y))
            end
        end
        return s
    end
    return maximum(smax, vcat(complex.(range(-beta, 0; length = 4L), δ),
                              complex.(-beta, range(0, δ; length = 2L)),
                              complex.(0.0, range(0, δ; length = 2L))))
end

# Splits the stage-list weights into per-chain vectors indexed like the stepping loop
# (entry j multiplies f(Y_{j-1}) of that chain; branch entries ≤ share+1 are unused), and
# returns each chain's stage times (row sums of the Butcher matrix).
function per_chain(w, stages, pos, A, chs, L, k)
    bs = [zeros(L) for _ in chs]
    ts = [zeros(L) for _ in chs]
    csum = vec(sum(A; dims = 2))
    for c in eachindex(chs), j in 0:(L - 1)
        ts[c][j + 1] = csum[pos(c, j)]
        (c == 1 || j > k) && (bs[c][j + 1] = w[pos(c, j)])
    end
    return bs, ts
end

###############################################################################
# Refinement of the chosen design. This includes optimal weights (a linear program) and a short
# continuous adjustment of the chain parameters, both aimed at lower internal amplification.
# It keeps the chosen shared stages, so the RHS cost is unchanged, and no next-order error
# term may grow, so accuracy is never worse.

# The design problem's matrices for given chains and shared stages. order and stability
# conditions (Aeq·w = beq), next-order error rows (E·w ≈ e) and amplification rows (Rm·w,
# one per sample point and perturbed stage).
function design_matrices(chs, L, k, target, beta, kids, ord, gamma; npts = 30)
    stages, pos, A, polys = branched_system(chs, L, k)
    S = length(stages)
    xs = [-beta * (1 - cos((2i - 1) * pi / (2L))) / 2 for i in 1:L]
    stab = reduce(vcat, [transpose(x .* polys(x)) for x in xs])
    G = Vector{Vector{Float64}}(undef, length(kids)); AG = similar(G)
    for i in eachindex(kids)
        g = ones(S)
        for kk in kids[i]
            g .*= AG[kk]
        end
        G[i] = g; AG[i] = A * g
    end
    nt = findall(i -> !istall(kids, i) && ord[i] <= ORDER, eachindex(kids))
    tn = findall(==(ORDER + 1), ord)
    Aeq = vcat(stab, reduce(vcat, [transpose(G[t]) for t in nt]))
    beq = vcat([target(x) - 1 for x in xs], 1 ./ gamma[nt])
    E = reduce(vcat, [transpose(G[t]) for t in tn]); e = 1 ./ gamma[tn]
    Rm = zeros(npts * S, S); P = zeros(S); r = 0
    for x in range(-beta, 0; length = npts), s0 in 1:S
        fill!(P, 0.0); P[s0] = 1.0
        for (s, (c, j)) in enumerate(stages)
            (s <= s0 || j == 0) && continue
            ch = chs[c]
            P[s] = j == 1 ? P[pos(c, 0)] * (1 + ch.mt[1] * x) :
                   ch.mu[j] * P[pos(c, j - 1)] + ch.ka[j] * P[pos(c, j - 2)] + ch.mt[j] * x * P[pos(c, j - 1)]
        end
        r += 1; Rm[r, :] = x .* P
    end
    return Aeq, beq, E, e, Rm, stages, pos, A
end

# Weights with the smallest amplification. we minimize t subject to |Rm·w| ≤ t, the exact
# conditions, and |E·w − e| ≤ δ tree by tree. Returns (t, w), or (Inf, nothing)
function lp_weights(Aeq, beq, E, e, Rm, δ)
    MOI = HiGHS.MOI
    model = HiGHS.Optimizer(); MOI.set(model, MOI.Silent(), true)
    S = size(Aeq, 2)
    w = MOI.add_variables(model, S); t = MOI.add_variable(model)
    aff(row) = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[MOI.ScalarAffineTerm(row[j], w[j]) for j in 1:S if row[j] != 0], 0.0)
    for i in axes(Aeq, 1)
        MOI.add_constraint(model, aff(Aeq[i, :]), MOI.EqualTo(beq[i]))
    end
    for i in axes(E, 1)
        MOI.add_constraint(model, aff(E[i, :]), MOI.Interval(e[i] - δ[i], e[i] + δ[i]))
    end
    tt = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, t)], 0.0)
    for i in axes(Rm, 1)
        f = aff(Rm[i, :])
        MOI.add_constraint(model, MOI.Utilities.operate(-, Float64, f, tt), MOI.LessThan(0.0))
        MOI.add_constraint(model, MOI.Utilities.operate(+, Float64, f, tt), MOI.GreaterThan(0.0))
    end
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), tt)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL || return (Inf, nothing)
    return MOI.get(model, MOI.VariablePrimal(), t), MOI.get.(model, MOI.VariablePrimal(), w)
end

# Short Nelder–Mead on the chain parameters with the linear programming inside to find 
# the best refinment to the design
function refine_design(chosen, L, beta, target, kids, ord, gamma; iters = 60)
    k = chosen.share
    nc = length(chosen.chs)
    # no next-order error term may grow beyond its value in the chosen design
    _, _, E0, e0, _ = design_matrices(chosen.chs, L, k, target, beta, kids, ord, gamma; npts = 2)
    δ = abs.(E0 * chosen.w .- e0) .+ 1e-12
    θ0 = reduce(vcat, [[log(c.nu), log(max((c.v0 - 1) * L^2, 1e-3)),
                        log(max((1 + c.v0) / (c.v1 * beta) - 1, 1e-3))] for c in chosen.chs])
    mk(θ) = [chain(L, exp(θ[3c - 2]), exp(θ[3c - 1]), beta, 1 + exp(θ[3c])) for c in 1:nc]
    function f(θ)
        all(c -> 0.15 <= exp(θ[3c - 2]) <= 8 && exp(θ[3c - 1]) <= 500 && exp(θ[3c]) <= 3, 1:nc) || return Inf
        Aeq, beq, E, e, Rm = design_matrices(mk(θ), L, k, target, beta, kids, ord, gamma)
        return first(lp_weights(Aeq, beq, E, e, Rm, δ))
    end
    n = length(θ0)                                      
    X = [copy(θ0) for _ in 1:(n + 1)]
    for i in 1:n
        X[i + 1][i] += 0.3
    end
    F = f.(X)
    for _ in 1:iters
        p = sortperm(F); X = X[p]; F = F[p]
        xc = sum(X[1:n]) / n; xr = xc + (xc - X[end]); fr = f(xr)
        if fr < F[1]
            xe = xc + 2(xc - X[end]); fe = f(xe)
            X[end], F[end] = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < F[n]
            X[end], F[end] = xr, fr
        else
            xk = xc + 0.5(X[end] - xc); fk = f(xk)
            if fk < F[end]
                X[end], F[end] = xk, fk
            else
                for i in 2:(n + 1)
                    X[i] = X[1] + 0.5(X[i] - X[1]); F[i] = f(X[i])
                end
            end
        end
    end
    chs = mk(X[argmin(F)])
    Aeq, beq, E, e, Rm, stages, pos, A = design_matrices(chs, L, k, target, beta, kids, ord, gamma; npts = 60)
    amp, w = lp_weights(Aeq, beq, E, e, Rm, δ)
    w === nothing && return nothing
    w += pinv(Aeq; rtol = 1e-13) * (beq - Aeq * w)        # polish the exact conditions
    return (; share = k, chs, w, errp = norm(E * w - e), stages, pos, A,
            amp = output_amplification_shared(chs, L, k, w, beta))
end

###############################################################################
# Design the method (quasi-Butcher tableau). this is computed once and then cached forever

# The chain variants to choose from when building our polynomial combination.
const CHAIN_SPECS = [(nu, eps, s) for nu in (0.5, 1.0, 2.0, 3.0) for eps in (0.5, 2.0, 10.0, 30.0)
                     for s in (1.0, 1.5, 2.0)]
# The original variants (coverage 1); the best of them sets the accuracy bar. This is the trunk chain, the one we must evaluate in full.
const BASE_SPECS = [(nu, eps, 1.0) for nu in (0.5, 1.0, 2.0) for eps in (0.5, 2.0, 10.0, 30.0)]

# File stores numbers only; rebuild Chain on load. ERR_KEY is :err4 (order 3) or :err5 (order 4).
load_cache(path) = begin
    m = deserialize(path)
    chains = [Chain(c.nu, c.v0, c.v1, c.mu, c.ka, c.mt, c.c) for c in m.chains]
    merge((; m.M, m.L, m.nu, m.damp, m.beta, chains, m.b, m.b_hat, m.times, m.share),
          NamedTuple{(ERR_KEY,)}((getproperty(m, ERR_KEY),)),
          (; m.amp, m.amp_min, m.rhs_per_step))
end
save_cache(path, m) = serialize(path, merge(
    (; m.M, m.L, m.nu, m.damp, m.beta,
     chains = [(nu = c.nu, v0 = c.v0, v1 = c.v1, mu = c.mu, ka = c.ka, mt = c.mt, c = c.c) for c in m.chains],
     m.b, m.b_hat, m.times, m.share),
    NamedTuple{(ERR_KEY,)}((getproperty(m, ERR_KEY),)),
    (; m.amp, m.amp_min, m.rhs_per_step)))

"""
    design(M; nu = 0.5, damp = DAMP, nchains = NCHAINS_DEFAULT, kappa = 0.05, npool = 20)
    design(M; amp_factor, ...)

Runtime loads from `default/`. Pass `amp_factor` only for offline precompute, which reads/writes the named variant folder.
`amp_factor` is how much internal amplification (relative to the lowest found) may be traded for shared stages.
"""
function design(M; nu = 0.5, damp = DAMP, nchains = NCHAINS_DEFAULT, kappa = 0.05,
                amp_factor = nothing, npool = 20)
    path = amp_factor === nothing ? cache_path(M) : variant_cache_path(M; nu, damp, amp_factor)
    af = amp_factor === nothing ? 5.0 : amp_factor
    isfile(path) && return load_cache(path)
    m = design_uncached(M; nu, damp, nchains, kappa, amp_factor = af, npool)
    mkpath(dirname(path))
    save_cache(path, m)
    return m
end

#if we ever dont wanna cache it we call this. this does the actual design.
function design_uncached(M; nu = 0.5, damp = DAMP, nchains = NCHAINS_DEFAULT, kappa = 0.05,
                         amp_factor = 5.0, npool = 20, nsample = 10000)
    L = ORDER * M
    beta = find_beta(nu, M; damp)
    w0 = 1 + damp / M^2
    d0, d = order_coeffs(nu, M, beta, w0)
    R(x) = evalG(nu, M, d0, d, w0 + (1 + w0) * x / beta)
    kids, ord, gamma = trees(ORDER + 1)

    # Step 1: every combo of chain variants (separate chains). Accuracy bar from the
    # original variants; lowest amp among designs at least that accurate is the reference.
    cands = []
    # All combinations when there are few (orders 3 and 4); otherwise every combination of the
    # original variants (so the accuracy bar exists) plus a reproducible random sample.
    combos = if binomial(length(CHAIN_SPECS), nchains) <= 20_000
        subsets(collect(eachindex(CHAIN_SPECS)), nchains)
    else
        rng = MersenneTwister(1)
        basei = findall(sp -> sp in BASE_SPECS, CHAIN_SPECS)
        unique([subsets(basei, nchains);
                [sort(randperm(rng, length(CHAIN_SPECS))[1:nchains]) for _ in 1:nsample]])
    end
    for idx in combos
        specs = CHAIN_SPECS[idx]
        chs = [chain(L, sp[1], sp[2], beta, sp[3]) for sp in specs]
        bs, res, errp = solve_weights(chs, L, R, ORDER, beta, kids, ord, gamma)
        res > 1e-10 && continue
        push!(cands, (amp = output_amplification(chs, bs, L, beta; npts = 40), errp, chs, specs))
    end
    isempty(cands) && error("design failed for M=$M")
    base = filter(c -> all(sp -> sp in BASE_SPECS, c.specs), cands)
    bar = isempty(base) ? Inf : argmin(c -> c.amp * c.errp, base).errp
    good = sort(filter(c -> c.errp <= bar, cands); by = c -> c.amp)
    amp_min = good[1].amp
    cap = amp_factor * amp_min

    # Step 2: most shared stages. Default: largest share, lowest amp.
    # CRKG4 M=3 only: among shares whose RHS is within 20% of the max-share cost, minimize amp × stage on the box
    # (hyperbolic CFL picks M=3; that design otherwise has large imag-axis stage growth).
    pool = good[1:min(npool, length(good))]
    chosen = nothing
    stage_trade = (ORDER == 4 && M == 3)
    δbox = stage_trade ? find_delta(nu, M, beta; damp) : 0.0
    rhs_of(share) = L + (nchains - 1) * (L - 1 - share)
    top = nothing
    for share in (L - 2):-1:0
        stage_trade && top !== nothing && rhs_of(share) > 1.2 * rhs_of(top) && break
        for cand in pool, tr in 1:nchains
            chs = cand.chs[[tr; setdiff(1:nchains, tr)]]
            w, res, errp, stages, pos, A = solve_weights_shared(chs, L, share, R, ORDER, beta, kids, ord, gamma)
            (res > 1e-10 || errp > bar) && continue
            amp = output_amplification_shared(chs, L, share, w, beta; npts = 40)
            amp > cap && continue
            if stage_trade
                top === nothing && (top = share)
                score = amp * stage_box_max(chs, L, share, beta, δbox)
                (chosen === nothing || score < chosen.score) &&
                    (chosen = (; share, chs, w, errp, amp, stages, pos, A, score))
            else
                (chosen === nothing || amp < chosen.amp) &&
                    (chosen = (; share, chs, w, errp, amp, stages, pos, A))
            end
        end
        !stage_trade && chosen !== nothing && break
    end
    if chosen === nothing          # nothing within the cap: best chains, share only f(u)
        chs = pool[1].chs
        w, res, errp, stages, pos, A = solve_weights_shared(chs, L, 0, R, ORDER, beta, kids, ord, gamma)
        chosen = (share = 0, chs = chs, w = w, errp = errp,
                  amp = output_amplification_shared(chs, L, 0, w, beta; npts = 40), stages, pos, A)
    end
    # now refines our design to find the best weights and chain parameters
    refined = refine_design(chosen, L, beta, R, kids, ord, gamma)
    if refined !== nothing
        if stage_trade
            refined.amp * stage_box_max(refined.chs, L, refined.share, beta, δbox) <
                chosen.amp * stage_box_max(chosen.chs, L, chosen.share, beta, δbox) && (chosen = refined)
        else
            refined.amp < chosen.amp && (chosen = refined)
        end
    end
    share, chs = chosen.share, chosen.chs

    # Embedded weights: nearby order-(ORDER-1) polynomial + order conditions up to ORDER-1.
    Rh = nothing
    for kp in kappa .* [1, -1, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1]
        dh0, dh = order_coeffs(nu, M, beta, w0; p = ORDER - 1, dN = d[ORDER] * (1 + kp))
        f(x) = evalG(nu, M, dh0, dh, w0 + (1 + w0) * x / beta)
        if all(abs(f(x)) <= 1.02 for x in range(-beta, 0; length = 40L))
            Rh = f
            break
        end
    end
    Rh === nothing && error("no stable embedded polynomial for M=$M")
    wh, res, _ = solve_weights_shared(chs, L, share, Rh, ORDER - 1, beta, kids, ord, gamma)
    res > 1e-10 && error("embedded weights failed for M=$M")

    bs, times = per_chain(chosen.w, chosen.stages, chosen.pos, chosen.A, chs, L, share)
    bhs, _ = per_chain(wh, chosen.stages, chosen.pos, chosen.A, chs, L, share)
    return merge((; M, L, nu, damp, beta, chains = chs, b = bs, b_hat = bhs, times, share),
                 NamedTuple{(ERR_KEY,)}((chosen.errp,)),
                 (; amp = output_amplification_shared(chs, L, share, chosen.w, beta),
                  amp_min, rhs_per_step = L + (nchains - 1) * (L - 1 - share)))
end
