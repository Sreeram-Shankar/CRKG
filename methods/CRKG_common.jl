# Shared CRKG stepping utilities. Included by CRKG3.jl / CRKG4.jl (same module).

###############################################################################
# Stepping

const SERIAL_BELOW = 1 << 15
@inline nchunks(n) = n < SERIAL_BELOW ? 1 : Threads.nthreads()
@inline chunk(c, n, nc) = (div((c - 1) * n, nc) + 1):div(c * n, nc)

# also does NUMA first touch: each thread places the pages it will use later
function tfill!(v, x)
    n = length(v); nc = nchunks(n); xv = convert(eltype(v), x)
    @batch for c in 1:nc
        @inbounds @simd for i in chunk(c, n, nc); v[i] = xv; end
    end
    return v
end

function tcopy!(Y, u)
    n = length(Y); nc = nchunks(n)
    @batch for c in 1:nc
        @inbounds @simd for i in chunk(c, n, nc); Y[i] = convert(eltype(Y), u[i]); end
    end
    return Y
end

# Working buffers for one step. Ym/Y are the two latest stages (Float32 when low), F is
# the RHS (empty when low, then F32 is used instead), s is the F64 update, SF holds the
# shared RHS. Adaptive runs also get a Float32 e which is the emebed error. branch=true allocates Sk/Skm as
# Float32 diffs from u_n. essentially, this manages storage and ensures that the storage of the method is good
# and also handles mixed precision
function workspace(m, n, ::Type{T} = Float64; low = false, emb = true,
                   branch = m.share > 0) where {T <: AbstractFloat}
    snap = branch ? n : 0
    Ts = T === Float64 ? Float32 : T
    Ty = low ? Float32 : T
    z(S, len) = tfill!(Vector{S}(undef, len), zero(S))
    return (Ym = z(Ty, n), Y = z(Ty, n), Yt = z(Ty, n), F = z(T, low ? 0 : n), s = z(T, n),
            e = z(Ts, emb ? n : 0), F32 = z(Float32, low ? n : 0),
            Sk = z(Ts, snap), Skm = z(Ts, snap), SF = z(Ty, n),
            red = zeros(Float64, Threads.nthreads()))
end

# updates the step result and teh embedded s, so this does the 
# rhs sum while the one blow does the update. first stage is a plain forward Euler step
function k_euler!(s, e, Yn, Y, F, hb, hdb, hm, emb::Bool)   # first stage
    n = length(Y); nc = nchunks(n); hmT = convert(eltype(Yn), hm)
    if emb
        hdbE = convert(eltype(e), hdb)
        @batch for c in 1:nc
            @inbounds @simd for i in chunk(c, n, nc)
                fi = F[i]; s[i] += hb * fi; e[i] += hdbE * fi; Yn[i] = Y[i] + hmT * fi
            end
        end
    else
        @batch for c in 1:nc
            @inbounds @simd for i in chunk(c, n, nc)
                fi = F[i]; s[i] += hb * fi; Yn[i] = Y[i] + hmT * fi
            end
        end
    end
end

# the three-term recurrence step for the later stages of the chain
function k_stage!(s, e, Yn, Y, Ym, F, hb, hdb, mu, ka, hm, emb::Bool)   # later stages
    n = length(Y); nc = nchunks(n); Ty = eltype(Yn)
    muT, kaT, hmT = convert(Ty, mu), convert(Ty, ka), convert(Ty, hm)
    if emb
        hdbE = convert(eltype(e), hdb)
        @batch for c in 1:nc
            @inbounds @simd for i in chunk(c, n, nc)
                fi = F[i]; s[i] += hb * fi; e[i] += hdbE * fi
                Yn[i] = muT * Y[i] + kaT * Ym[i] + hmT * fi
            end
        end
    else
        @batch for c in 1:nc
            @inbounds @simd for i in chunk(c, n, nc)
                fi = F[i]; s[i] += hb * fi
                Yn[i] = muT * Y[i] + kaT * Ym[i] + hmT * fi
            end
        end
    end
end

function k_acc!(s, e, F, hb, hdb, emb::Bool)   # last stage of a chain
    n = length(s); nc = nchunks(n)
    if emb
        hdbE = convert(eltype(e), hdb)
        @batch for c in 1:nc
            @inbounds @simd for i in chunk(c, n, nc)
                fi = F[i]; s[i] += hb * fi; e[i] += hdbE * fi
            end
        end
    else
        @batch for c in 1:nc
            @inbounds @simd for i in chunk(c, n, nc); s[i] += hb * F[i]; end
        end
    end
end

function k_snap!(SF, Sk, Skm, F, Y, Ym, u)
    n = length(Y); nc = nchunks(n)
    @batch for c in 1:nc
        @inbounds @simd for i in chunk(c, n, nc)
            ui = convert(eltype(Y), u[i])
            SF[i] = F[i]; Sk[i] = Y[i] - ui; Skm[i] = Ym[i] - ui
        end
    end
end

function k_restart!(Y, Ym, u, Sk, Skm, SF, mu, ka, hm)   # share > 0
    n = length(Y); nc = nchunks(n); Ty = eltype(Y)
    muT, kaT, hmT = convert(Ty, mu), convert(Ty, ka), convert(Ty, hm)
    @batch for c in 1:nc
        @inbounds @simd for i in chunk(c, n, nc)
            ui = convert(eltype(Y), u[i])
            a = ui + convert(eltype(Y), Sk[i]); b = ui + convert(eltype(Y), Skm[i])
            Ym[i] = a; Y[i] = muT * a + kaT * b + hmT * SF[i]
        end
    end
end

function k_restart0!(Y, Ym, u, SF, hm)   # share = 0
    n = length(Y); nc = nchunks(n); hmT = convert(eltype(Y), hm)
    @batch for c in 1:nc
        @inbounds @simd for i in chunk(c, n, nc)
            ui = convert(eltype(Y), u[i]); Ym[i] = ui; Y[i] = ui + hmT * SF[i]
        end
    end
end

function k_final!(u, s, commit::Bool)
    n = length(u); nc = nchunks(n)
    if commit
        @batch for c in 1:nc
            @inbounds @simd for i in chunk(c, n, nc); u[i] += s[i]; end
        end
    else
        @batch for c in 1:nc
            @inbounds @simd for i in chunk(c, n, nc); s[i] += u[i]; end
        end
    end
end

function k_final_emb!(red, u, s, e, atol, rtol, commit::Bool)
    n = length(u); nc = nchunks(n)
    @batch for c in 1:nc
        acc = 0.0
        @inbounds for i in chunk(c, n, nc)
            ui = u[i]; un = ui + s[i]; ei = convert(eltype(u), e[i])
            atol !== nothing && (ei /= atol + rtol * max(abs(ui), abs(un)))
            acc += abs2(ei)
            commit ? (u[i] = un) : (s[i] = un)
        end
        red[c] = acc
    end
    est = 0.0
    @inbounds for c in 1:nc; est += red[c]; end
    return est
end

# RHS for stages after the first. If rhs_low! is given we evaluate in Float32 into F32
@inline function stage_rhs!(ws, rhs!, rhs_low!, Y, tj)
    if rhs_low! !== nothing
        rhs_low!(ws.F32, Y, Float32(tj))
        return ws.F32
    end
    rhs!(ws.F, Y, tj)
    return ws.F
end

# One CRKG step if the adaptive dt is used (or the embedded error is on). Resizes the workspace if n changed (mesh adapt), then dispatches on
# Val(emb) (eheter the embedded error or not is on) so the embedded accumulation compiles out when emb=false and storage is saved for later use.
    #the reason we dispatch is for storage optimization and to avoid code duplication
function step!(rhs!, limiter!, u::AbstractVector{T}, h::T, t, m, ws;
               atol = nothing, rtol = nothing, commit = true, emb = true,
               rhs_low! = nothing) where {T <: AbstractFloat}
    n = length(u)
    if length(ws.Y) != n
        for (name, v) in pairs(ws)
            name === :red || isempty(v) || resize!(v, n)
        end
    end
    return step!(rhs!, limiter!, u, h, t, m, ws, atol, rtol, commit, rhs_low!, Val(emb))
end

# The actual step. with a flaot32 rhs_low!, Y is in float32, and we cast u to Y once, then every RHS sees
# Float32 state. this is the mixed precision part of the method. This uses the normal gegenbaur/stabilized RK method
# internal stages to take the supertimestep 
function step!(rhs!, limiter!, u::AbstractVector{T}, h::T, t, m, ws, atol, rtol, commit,
               rhs_low!, ::Val{emb}) where {T <: AbstractFloat, emb}
    Ym, Y, Yt, s, e, Sk, Skm, SF = ws.Ym, ws.Y, ws.Yt, ws.s, ws.e, ws.Sk, ws.Skm, ws.SF
    tfill!(s, 0)
    emb && tfill!(e, 0)
    L, k, chains, bs, bhs, times = m.L, m.share, m.chains, m.b, m.b_hat, m.times
    tr = chains[1]

    #we perform the trunk chain first, then the branch chains.

    # for the first stage of the trunk, cast u into Y once when low, then f(Y,t) in that precision
    tcopy!(Y, u)
    rhs_low! !== nothing ? rhs_low!(SF, Y, Float32(t)) : rhs!(SF, u, t)
    k_euler!(s, e, Yt, Y, SF, h * bs[1][1], emb ? h * (bs[1][1] - bhs[1][1]) : zero(h),
             h * tr.mt[1], emb)
    Ym, Y, Yt = Y, Yt, Ym

    #this is an important part of the method, every internal stage of the method is a valid 
    #approximation of the solution, 
    limiter! !== nothing && limiter!(Y, t + times[1][2] * h)

    # now goes through the later stages of the trunk, keeping the same precision as the first stage
    for j in 2:L
        F = stage_rhs!(ws, rhs!, rhs_low!, Y, t + times[1][j] * h)
        j == k + 1 && k_snap!(SF, Sk, Skm, F, Y, Ym, u)
        hb, hdb = h * bs[1][j], emb ? h * (bs[1][j] - bhs[1][j]) : zero(h)
        if j == L
            k_acc!(s, e, F, hb, hdb, emb); break
        end
        k_stage!(s, e, Yt, Y, Ym, F, hb, hdb, tr.mu[j], tr.ka[j], h * tr.mt[j], emb)
        Ym, Y, Yt = Y, Yt, Ym
        limiter! !== nothing && limiter!(Y, t + times[1][j + 1] * h)
    end

    #the previous loop was for the trunk. now, we apply it to the other chains, or the branches. 
    # because some of these chains can share rhs computations, we handle this after the trunk 
    #and continue using the mixed precision from the trunk.
    for c in 2:length(chains)
        ch = chains[c]

        #k is the amount of rhs/stages shared between the branches and the trunk. if there are shared ones, we reuse them from the trunk.
        k > 0 ? k_restart!(Y, Ym, u, Sk, Skm, SF, ch.mu[k + 1], ch.ka[k + 1], h * ch.mt[k + 1]) :
                k_restart0!(Y, Ym, u, SF, h * ch.mt[1])
        limiter! !== nothing && limiter!(Y, t + times[c][k + 2] * h)

        #finish off the branches after checking for shared stages
        for j in (k + 2):L
            F = stage_rhs!(ws, rhs!, rhs_low!, Y, t + times[c][j] * h)
            hb, hdb = h * bs[c][j], emb ? h * (bs[c][j] - bhs[c][j]) : zero(h)
            if j == L
                k_acc!(s, e, F, hb, hdb, emb); break
            end
            k_stage!(s, e, Yt, Y, Ym, F, hb, hdb, ch.mu[j], ch.ka[j], h * ch.mt[j], emb)
            Ym, Y, Yt = Y, Yt, Ym
            limiter! !== nothing && limiter!(Y, t + times[c][j + 1] * h)
        end
    end

    #now we compute the embedded error if needed and update the solution
    est = emb ? T(k_final_emb!(ws.red, u, s, e, atol, rtol, commit)) :
                (k_final!(u, s, commit); zero(T))
    limiter! !== nothing && limiter!(commit ? u : s, t + h)
    return emb ? (atol !== nothing ? sqrt(est / length(u)) : sqrt(est)) : zero(T)
end

# Remembers interval lengths so picking M each step is cheap when dt changes.
#also saves the method tableaus for the given M, nu, and damp so its cheap to keep (always from default/)
const length_cache = Dict{Tuple{Float64,Float64,Int},Float64}()
const imag_cache = Dict{Tuple{Float64,Float64,Int},Float64}()
const method_cache = Dict{Tuple{Int,Float64,Float64},Any}()

# Half-height of the largest box [-beta, 0] × [-δ, δ] inside |R| <= 1, which the CFL controller relies on
function find_delta(nu, M, beta; damp = DAMP)
    get!(imag_cache, (Float64(nu), Float64(damp), M)) do
        w0 = 1 + damp / M^2
        d0, d = order_coeffs(nu, M, beta, w0)
        stable(x, y) = abs(evalG(nu, M, d0, d, w0 + (1 + w0) * complex(x, y) / beta)) <= 1 + 1e-8
        δ = 40.0
        for x in range(-beta, 0.0; length = 401)
            y = 0.0
            while y + 0.01 <= δ && stable(x, y + 0.01)
                y += 0.01
            end
            δ = min(δ, y)
        end
        δ
    end
end

# Cheapest (M, dt). Dissipation of both parts sits on the real axis. The imaginary cap uses
# ω = wave_fraction·ρ_h. wave_fraction defaults to 1 (the whole spectral radius).
# Safeties are in (0, 1]. Called again every step.
function cfl_controller(ρ_h, ρ_p; Mmax = 30, nu = 0.5, damp = DAMP,
                        safety_hyperbolic = 1.0, safety_parabolic = 1.0, wave_fraction = 1.0)
    best, M_, h = Inf, 2, 0.0
    ω = wave_fraction * ρ_h
    for M in 2:Mmax
        m = get!(method_cache, (M, Float64(nu), Float64(damp))) do
            design(M; nu, damp)
        end
        δ = find_delta(nu, M, m.beta; damp)
        # NaN radii must not become Inf (Julia: NaN > 0 is false → old branch returned Inf).
        hr = !(isfinite(ρ_p) && isfinite(ρ_h)) ? 0.0 :
             (ρ_p + ρ_h > 0 ? safety_parabolic * m.beta / (ρ_p + ρ_h) : Inf) # real extent of both parts together, which add
        hw = !isfinite(ω) ? 0.0 :
             (ω > 0 ? safety_hyperbolic * δ / ω : Inf)                        # imaginary extent, inside the box by construction of δ
        hi = min(hr, hw)
        work = m.rhs_per_step / hi
        work < best && (best = work; M_ = M; h = hi)
    end
    return M_, h
end

#power iteration to estimate the spectral radius of the Jacobian of the rhs
function estimate_rho(rhs!, u, t)
    v = randn(length(u)); v ./= norm(v)
    du, w, up = similar(u), similar(u), similar(u)
    ρ = 0.0
    for _ in 1:30 #30 iterations is enough for most problems
        @. up = u + 1e-8 * v
        rhs!(du, up, t); rhs!(w, u, t)
        @. du = (du - w) / 1e-8
        ρ = norm(du)
        v .= du ./ (ρ + 1e-30)
    end
    return ρ
end

# solve CRKG in place
function solve!(rhs!, u, tspan, dt = nothing, M = nothing;
                limiter! = nothing, rhs_low! = nothing, callback = nothing, Mmax = 50,
                #limiter! is the limiter function, rhs_low! is the flaot32 rhs used when you want mixed precision. These 2 are important
                #callback is a callback, and Mmax is the maximum number of stages. 
                rho = nothing, adaptive = false, #rho is the eigenvalue estimatee which is power iteration by default. adaptive is just a boolean for the adaptive step size.
                #it can also be :adaptive_error_based or :cfl_based. the former is standard embeded control. :cfl_based takes rho_hyperbolic and rho_parabolic and picks M and dt each step.
                rho_hyperbolic = nothing, rho_parabolic = nothing,
                safety_hyperbolic = 1.0, safety_parabolic = 1.0, #in (0, 1], scale the hyperbolic and parabolic step allowances
                wave_fraction = 1.0, #max|Im λ| / ρ_h. Dissipation uses β, waves use δ. Default 1 uses the whole spectral radius.
                atol = 1e-6, rtol = 1e-6,
                β1 = 0.7, β2 = 0.4, β3 = 0.0, safety = 0.9, facmin = 0.2, facmax = 2.0, dtmax = Inf, #these are control parameters for the adaptive step size. our opinion
                #is that adaptive stepping veers from numerics into control theory so we just use whatever the standard is
                nu = 0.5, damp = DAMP) #we do not recommend changing these (gegenbaur parameter, endpoint shift) but you can if you want. tableaus always come from default/
                
    t, tend = tspan

    #the eigenvalue estimate can be provided dependant on the problem. if not provided, we estimate it using power iteration
    ρ(t_, u_) = rho === nothing ? estimate_rho(rhs!, u_, t_) :
                rho isa Number ? Float64(rho) : Float64(rho(t_, u_))

    # fixed step size mode
    if adaptive === false
        (M === nothing && dt === nothing) && error("solve!: fixed step needs M or dt") #one of M or dt must be provided
        if M !== nothing && dt !== nothing #if both are provided, we use the provided M and dt without any further calculations
            M_, h = Int(M), Float64(dt)
        else
            ρv = ρ(t, u)
            if dt !== nothing #if dt is provided, we use it to calculate the M. namely, the min M that covers the eigenvalue for the given dt
                h = Float64(dt)
                M_ = min_M(1.05 * h * ρv; nu, damp)
                if M_ > Mmax
                    M_ = Mmax
                    h = get!(length_cache, (Float64(nu), Float64(damp), Mmax)) do
                        find_beta(nu, Mmax; damp)
                    end / (1.05 * ρv)
                end
            else
                M_ = Int(M) #if M is provided, we use it to calculate the h. namely, the smallest dt that covers the eigenvalue for the given M
                h = get!(length_cache, (Float64(nu), Float64(damp), M_)) do
                    find_beta(nu, M_; damp)
                end / (1.05 * ρv)
            end
        end

        #we get the method tableau and cache it for the given M
        m = get!(method_cache, (M_, Float64(nu), Float64(damp))) do
            design(M_; nu, damp)
        end

        #build the worksapce based on the method tableau and the number of equations
        ws = workspace(m, length(u); low = rhs_low! !== nothing, emb = false, branch = true)

        #we now step through the solution. naccept is just bookeeping since this is fixed dt
        naccept = 0
        while t < tend - 10 * eps()
            hi = min(h, tend - t)
            step!(rhs!, limiter!, u, hi, t, m, ws; emb = false, rhs_low!)
            t += hi; naccept += 1
            callback !== nothing && callback(t, u)
        end
        return (accepted = naccept, rejected = 0, dt_last = h, M = M_)
    end

    # cfl controller. rho_hyperbolic and rho_parabolic are a number or (t, u) -> ρ. M and dt are recomputed every step.
    if adaptive === :cfl_based
        (rho_hyperbolic === nothing || rho_parabolic === nothing) && error("solve!: cfl_based needs rho_hyperbolic and rho_parabolic")
        ρh(t_, u_) = rho_hyperbolic isa Number ? Float64(rho_hyperbolic) : Float64(rho_hyperbolic(t_, u_))
        ρp(t_, u_) = rho_parabolic isa Number ? Float64(rho_parabolic) : Float64(rho_parabolic(t_, u_))
        ws = workspace(get!(method_cache, (2, Float64(nu), Float64(damp))) do
                           design(2; nu, damp)
                       end, length(u); low = rhs_low! !== nothing, emb = false, branch = true)
        naccept = 0
        h, M_ = 0.0, 2
        while t < tend
            #simple cfl-based controller based on the current spectral radii estiamtion
            M_, h = cfl_controller(ρh(t, u), ρp(t, u); Mmax, nu, damp,
                                   safety_hyperbolic, safety_parabolic, wave_fraction)
            hi = min(h, dtmax, tend - t)
            m = method_cache[(M_, Float64(nu), Float64(damp))]
            step!(rhs!, limiter!, u, hi, t, m, ws; emb = false, rhs_low!)
            t += hi; naccept += 1
            callback !== nothing && callback(t, u)
        end
        return (accepted = naccept, rejected = 0, dt_last = h, M = M_)
    end

    # adaptive error based mode
    adaptive === :adaptive_error_based || error("solve!: unknown adaptive=$adaptive")
    dt === nothing && error("solve!: adaptive_error_based needs dt0")
    h = Float64(dt)

    #build the worksapce based, but M now varies
    ws = workspace(get!(method_cache, (2, Float64(nu), Float64(damp))) do
                       design(2; nu, damp)
                   end, length(u); low = rhs_low! !== nothing, emb = true, branch = true)

    #this is smilar to fixed step size in that it calculates the min M needed but it adapts the step size based on the embedded error
    #the default control parameters are set from above and come from standard adaptive methods. treat this almost like a black box,
    # we dont rly trust this and dont like to use it, its just there for completeness and showcasing the embedded pair
    e1 = e2 = 1.0
    naccept = nreject = 0
    while t < tend
        ρv = ρ(t, u)
        h = min(h, dtmax, tend - t)
        M_ = min_M(1.05 * h * ρv; nu, damp)
        if M_ > Mmax
            M_ = Mmax
            h = get!(length_cache, (Float64(nu), Float64(damp), Mmax)) do
                find_beta(nu, Mmax; damp)
            end / (1.05 * ρv)
        end
        h = min(h, tend - t)
        m = get!(method_cache, (M_, Float64(nu), Float64(damp))) do
            design(M_; nu, damp)
        end
        err = step!(rhs!, limiter!, u, h, t, m, ws;
                    atol, rtol, commit = false, emb = true, rhs_low!)
        if err <= 1
            u .= ws.s
            t += h; naccept += 1
            callback !== nothing && callback(t, u)
            e = max(err, 1e-10)
            h *= clamp(safety * e^(-β1 / ORDER) * e1^(β2 / ORDER) * e2^(-β3 / ORDER), facmin, facmax)
            e2, e1 = e1, e
        else
            nreject += 1
            h *= clamp(safety * err^(-1 / ORDER), facmin, 1.0)
        end
    end
    return (accepted = naccept, rejected = nreject, dt_last = h)
end
