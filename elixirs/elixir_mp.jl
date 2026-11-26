# GPU mixed-precision spedupo

using CUDA
using Printf

CUDA.allowscalar(false)

const METHODS_DIR = joinpath(@__DIR__, "..", "methods")
include(joinpath(METHODS_DIR, "CRKG3.jl")); using .CRKG3
include(joinpath(METHODS_DIR, "CRKG4.jl")); using .CRKG4
const MODULES = (CRKG3, CRKG4)

###############################################################################
# settings

const GRIDS = (2048, 4096, 8192)   # bandwidth-bound on L40S; 1024 was launch-bound
const a_adv = 1.0
const ν = 5e-3
const Mmax = 30
const NSTEPS = 80            # timed steps (same schedule for f64 and mixed)
const NWARM = 5              # untimed warm-up steps

###############################################################################
# fused 1-launch GPU kernels (explicit casts; emb unused here so omitted from device code)

@inline function launch1d!(ker, n, args...)
    threads = 256
    blocks = cld(n, threads)
    @cuda threads = threads blocks = blocks ker(args...)
    return nothing
end

function copy_convert_kernel!(Y, u)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Y)
        @inbounds Y[i] = eltype(Y)(u[i])
    end
    return nothing
end

function euler_kernel!(s, Yn, Y, F, hb, hm)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Y)
        @inbounds begin
            fi = F[i]
            s[i] = s[i] + eltype(s)(hb) * eltype(s)(fi)
            Yn[i] = Y[i] + eltype(Yn)(hm) * eltype(Yn)(fi)
        end
    end
    return nothing
end

function stage_kernel!(s, Yn, Y, Ym, F, hb, mu, ka, hm)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Y)
        @inbounds begin
            Ty = eltype(Yn)
            fi = F[i]
            s[i] = s[i] + eltype(s)(hb) * eltype(s)(fi)
            Yn[i] = Ty(mu) * Y[i] + Ty(ka) * Ym[i] + Ty(hm) * Ty(fi)
        end
    end
    return nothing
end

function acc_kernel!(s, F, hb)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(s)
        @inbounds s[i] = s[i] + eltype(s)(hb) * eltype(s)(F[i])
    end
    return nothing
end

function snap_kernel!(SF, Sk, Skm, F, Y, Ym, u)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Y)
        @inbounds begin
            ui = eltype(Y)(u[i])
            SF[i] = F[i]
            Sk[i] = Y[i] - ui
            Skm[i] = Ym[i] - ui
        end
    end
    return nothing
end

function restart_kernel!(Y, Ym, u, Sk, Skm, SF, mu, ka, hm)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Y)
        @inbounds begin
            Ty = eltype(Y)
            ui = Ty(u[i])
            a = ui + Ty(Sk[i])
            b = ui + Ty(Skm[i])
            Ym[i] = a
            Y[i] = Ty(mu) * a + Ty(ka) * b + Ty(hm) * SF[i]
        end
    end
    return nothing
end

function restart0_kernel!(Y, Ym, u, SF, hm)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Y)
        @inbounds begin
            Ty = eltype(Y)
            ui = Ty(u[i])
            Ym[i] = ui
            Y[i] = ui + Ty(hm) * SF[i]
        end
    end
    return nothing
end

function final_kernel!(u, s, commit)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(u)
        @inbounds commit ? (u[i] = u[i] + s[i]) : (s[i] = s[i] + u[i])
    end
    return nothing
end

###############################################################################
# GPU CRKG methods (dispatch into fused kernels; signatures match step!)

function add_gpu_methods!(Mod)
    CA = CuArray
    Core.eval(Mod, quote
        tfill!(v::$CA, x) = fill!(v, convert(eltype(v), x))
        function tcopy!(Y::$CA, u)
            Main.launch1d!(Main.copy_convert_kernel!, length(Y), Y, u)
            return Y
        end
        function k_euler!(s::$CA, e, Yn, Y, F, hb, hdb, hm, emb::Bool)
            Main.launch1d!(Main.euler_kernel!, length(Y), s, Yn, Y, F, hb, hm)
            return nothing
        end
        function k_stage!(s::$CA, e, Yn, Y, Ym, F, hb, hdb, mu, ka, hm, emb::Bool)
            Main.launch1d!(Main.stage_kernel!, length(Y), s, Yn, Y, Ym, F, hb, mu, ka, hm)
            return nothing
        end
        function k_acc!(s::$CA, e, F, hb, hdb, emb::Bool)
            Main.launch1d!(Main.acc_kernel!, length(s), s, F, hb)
            return nothing
        end
        function k_snap!(SF::$CA, Sk, Skm, F, Y, Ym, u)
            Main.launch1d!(Main.snap_kernel!, length(Y), SF, Sk, Skm, F, Y, Ym, u)
            return nothing
        end
        function k_restart!(Y::$CA, Ym, u, Sk, Skm, SF, mu, ka, hm)
            Main.launch1d!(Main.restart_kernel!, length(Y), Y, Ym, u, Sk, Skm, SF, mu, ka, hm)
            return nothing
        end
        function k_restart0!(Y::$CA, Ym, u, SF, hm)
            Main.launch1d!(Main.restart0_kernel!, length(Y), Y, Ym, u, SF, hm)
            return nothing
        end
        function k_final!(u::$CA, s, commit::Bool)
            Main.launch1d!(Main.final_kernel!, length(u), u, s, commit)
            return nothing
        end
    end)
end
foreach(add_gpu_methods!, MODULES)

function gpu_workspace(n, low)
    Ty = low ? Float32 : Float64
    z(S, len) = CUDA.zeros(S, len)
    return (Ym = z(Ty, n), Y = z(Ty, n), Yt = z(Ty, n), F = z(Float64, low ? 0 : n),
            s = z(Float64, n), e = z(Float32, 0), F32 = z(Float32, low ? n : 0),
            Sk = z(Float32, n), Skm = z(Float32, n), SF = z(Ty, n),
            red = zeros(Float64, 1))
end

###############################################################################
# RHS on device

function ad_kernel!(du, u, N, c, d)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= N && j <= N
        ip = ifelse(i == N, 1, i + 1); im = ifelse(i == 1, N, i - 1)
        jp = ifelse(j == N, 1, j + 1); jm = ifelse(j == 1, N, j - 1)
        @inbounds begin
            uc = u[i + (j - 1) * N]
            ue = u[ip + (j - 1) * N]; uw = u[im + (j - 1) * N]
            un = u[i + (jp - 1) * N]; us = u[i + (jm - 1) * N]
            du[i + (j - 1) * N] = d * (ue + uw + un + us - 4 * uc) - c * (ue - uw)
        end
    end
    return nothing
end

function make_rhs(N)
    h = 1 / N
    thr, blk = (16, 16), (cld(N, 16), cld(N, 16))
    function rhs!(du, u, t)
        T = eltype(u)
        @cuda threads = thr blocks = blk ad_kernel!(du, u, N, T(a_adv / (2h)), T(ν / h^2))
        return du
    end
    return rhs!
end

ρ_h(h) = abs(a_adv) / h
ρ_p(h) = 8ν / h^2

function initial_state(N)
    h = 1 / N
    u0 = Vector{Float64}(undef, N^2)
    @inbounds for j in 1:N, i in 1:N
        x, y = (i - 1) * h, (j - 1) * h
        u0[i + (j - 1) * N] = 1 + 0.1 * sin(2π * x) * sin(2π * y)
    end
    return u0
end

# exact PDE solution at time t on the grid (equals initial_state at t = 0), built on the host
function exact_state(N, t)
    h = 1 / N
    u = Vector{Float64}(undef, N^2)
    @inbounds for j in 1:N, i in 1:N
        x, y = (i - 1) * h, (j - 1) * h
        u[i + (j - 1) * N] = 1 + 0.1 * exp(-8π^2 * ν * t) * sin(2π * (x - a_adv * t)) * sin(2π * y)
    end
    return u
end

# bytes for one mode at size n (rough upper bound)
function bytes_for(n, low)
    # u,s always F64; stages Ty; F or F32; Sk,Skm F32; SF Ty
    Ty = low ? 4 : 8
    return 2 * 8 * n + 3 * Ty * n + (low ? 4 : 8) * n + 2 * 4 * n + Ty * n
end

###############################################################################
# fixed-(M,dt) GPU stepping: schedule chosen once on the host, then pure device work

function choose_schedule(Mod, h)
    M, dt = Mod.cfl_controller(ρ_h(h), ρ_p(h); Mmax)
    m = Mod.method_cache[(M, 0.5, Mod.DAMP, 5.0)]
    return M, dt, m
end

function run_steps!(Mod, u, ws, rhs!, rl, m, dt, nsteps)
    t = 0.0
    for _ in 1:nsteps
        Mod.step!(rhs!, nothing, u, dt, t, m, ws; emb = false, rhs_low! = rl)
        t += dt
    end
    return nothing
end

# capture one step (AD rhs ignores t; fixed dt → same stage coeffs every step), then time launches only
function make_step_exec(Mod, u, ws, rhs!, rl, m, dt)
    graph = CUDA.capture() do
        Mod.step!(rhs!, nothing, u, dt, 0.0, m, ws; emb = false, rhs_low! = rl)
    end
    return CUDA.instantiate(graph)
end

function time_exec_steps(exec, nsteps)
    return CUDA.@elapsed begin
        for _ in 1:nsteps
            CUDA.launch(exec)
        end
        CUDA.synchronize()
    end
end

# relative L∞ difference on device (no host copies; one scalar sync)
function rel_err_linf(u, uref)
    diff = mapreduce((a, b) -> abs(a - b), max, u, uref)
    nrm = maximum(abs, uref)
    return Float64(diff) / max(Float64(nrm), 1e-30)
end

###############################################################################
# drive

free0, total = CUDA.free_memory(), CUDA.total_memory()
@printf("GPU: %s\n", CUDA.name(CUDA.device()))
@printf("memory: %.2f / %.2f GiB free\n", free0 / 2^30, total / 2^30)
@printf("2D AD  a=%.1f  ν=%.1e  NSTEPS=%d  NWARM=%d  Mmax=%d\n", a_adv, ν, NSTEPS, NWARM, Mmax)
@printf("grids: %s\n", join(string.(GRIDS), ", "))

@printf("warming CRKG caches M=2…%d …\n", Mmax)
for Mod in MODULES
    Mod.cfl_controller(1.0, 1.0; Mmax)   # designs + find_delta, once
end
println("done\n")

@printf("%-6s %6s %7s %5s %9s %8s %10s %9s %9s %12s %12s\n",
        "method", "N", "mode", "M", "RHS", "GiB", "GPU [s]", "ms/step", "speedup", "err vs f64", "err vs exact")

for N in GRIDS
    n = N^2

    u0 = initial_state(N)
    h = 1 / N
    rhs! = make_rhs(N)
    @printf("N=%d  ρ_h=%.3e  ρ_p=%.3e  free=%.2f GiB\n",
            N, ρ_h(h), ρ_p(h), CUDA.free_memory() / 2^30)

    for Mod in MODULES
        name = string(nameof(Mod))
        M, dt, m = choose_schedule(Mod, h)
        nf_step = m.rhs_per_step
        gib_f64 = bytes_for(n, false) / 2^30
        gib_mix = bytes_for(n, true) / 2^30
        uex = CuArray(exact_state(N, NSTEPS * dt))   # exact solution at the end of the timed run

        # --- Float64: allocate → warm → capture step graph → time launches only ---
        u_f64 = CuArray(u0)
        ws_f64 = gpu_workspace(n, false)
        run_steps!(Mod, u_f64, ws_f64, rhs!, nothing, m, dt, NWARM)
        CUDA.synchronize()
        copyto!(u_f64, u0)
        CUDA.synchronize()
        exec_f64 = make_step_exec(Mod, u_f64, ws_f64, rhs!, nothing, m, dt)
        CUDA.synchronize()
        t64 = time_exec_steps(exec_f64, NSTEPS)
        uref = copy(u_f64)
        err_disc = rel_err_linf(uref, uex)   # discretization (space + time) error of the Float64 run
        u_f64 = exec_f64 = nothing
        ws_f64 = nothing
        GC.gc(false)
        CUDA.reclaim()

        # --- mixed: allocate → warm → capture → time launches → err vs uref on device ---
        u_mix = CuArray(u0)
        ws_mix = gpu_workspace(n, true)
        run_steps!(Mod, u_mix, ws_mix, rhs!, rhs!, m, dt, NWARM)
        CUDA.synchronize()
        copyto!(u_mix, u0)
        CUDA.synchronize()
        exec_mix = make_step_exec(Mod, u_mix, ws_mix, rhs!, rhs!, m, dt)
        CUDA.synchronize()
        tmix = time_exec_steps(exec_mix, NSTEPS)
        err = rel_err_linf(u_mix, uref)
        err_mix_exact = rel_err_linf(u_mix, uex)
        speedup = t64 / tmix

        @printf("%-6s %6d %7s %5d %9d %8.2f %10.3f %9.3f %8.2fx %12.3e %12.3e\n",
                name, N, :f64, M, nf_step * NSTEPS, gib_f64, t64,
                1e3 * t64 / NSTEPS, 1.0, 0.0, err_disc)
        @printf("%-6s %6d %7s %5d %9d %8.2f %10.3f %9.3f %8.2fx %12.3e %12.3e\n",
                name, N, :mixed, M, nf_step * NSTEPS, gib_mix, tmix,
                1e3 * tmix / NSTEPS, speedup, err, err_mix_exact)
        println()

        u_mix = uref = uex = exec_mix = nothing
        ws_mix = nothing
        GC.gc(false)
        CUDA.reclaim()
    end
end
