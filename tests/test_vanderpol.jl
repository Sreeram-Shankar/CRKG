# we measure the convergence of the CRKG methods of 3rd and 4th order
# at fixed M = 8 with a dt ladder.
# We use the Feagin12 method as a reference at a fine dt.

include(joinpath(@__DIR__, "..", "methods", "CRKG3.jl"))
include(joinpath(@__DIR__, "..", "methods", "CRKG4.jl"))

using OrdinaryDiffEq, OrdinaryDiffEqFeagin
using LinearAlgebra, Printf

const T = BigFloat
setprecision(BigFloat, 256)

M = 8
t_final = T(10)
y0 = T[2, 0]
mu = T(2)
hs = [T(1) / T(2)^k for k in 1:7]  # 0.5 … 7.8125e-3
designs = Dict(CRKG3 => CRKG3.design(M),
               CRKG4 => CRKG4.design(M))

#our van der pel rhs function, mu = 2. this is not
#extremely stiff, but is a nonlinear test case to see
#convergence on an ODE problem that does not depend
#on a spatial discretization
rhs!(du, u, t) = (du[1] = u[2]; du[2] = mu * (1 - u[1]^2) * u[2] - u[1]; du)

#the CRKG method, it simply solves the problem with fixed M and dt
function integrate_crkg(mod, m, h)
    y = copy(y0)
    ws = mod.workspace(m, length(y), T; low = false, emb = false)
    t = zero(T); tf = t_final; hh = T(h)
    while t < tf - eps(T) * 10
        hi = min(hh, tf - t)
        mod.step!(rhs!, nothing, y, hi, t, m, ws; emb = false)
        t += hi
    end
    return y
end

#a feagi12 refence at a fine dt.
yref = solve(ODEProblem((du, u, p, t) -> rhs!(du, u, t), copy(y0), (zero(T), t_final)),
             Feagin12(); dt = hs[end] / 4, adaptive = false, save_everystep = false,
             dense = false, maxiters = 10_000_000, verbose = DEVerbosity(SciMLLogging.None())).u[end]

#we now run the convergence test for each method
results = Dict{String, Any}()
for (name, mod) in (("CRKG3", CRKG3), ("CRKG4", CRKG4))
    println(name)
    m = designs[mod]
    errs = [Float64(norm(integrate_crkg(mod, m, h) .- yref)) for h in hs]
    orders = [log(errs[i] / errs[i + 1]) / log(Float64(hs[i] / hs[i + 1])) for i in 1:(length(errs) - 1)]
    results[name] = Dict("beta" => m.beta, "dt" => Float64.(hs), "err" => errs, "orders" => orders)
end

for name in ("CRKG3", "CRKG4")
    r = results[name]
    @printf("\n%s  beta=%.5g\n", name, r["beta"])
    @printf("%12s %12s %10s\n", "dt", "err", "order")
    for i in eachindex(r["dt"])
        ord = i < length(r["dt"]) ? @sprintf("%.4f", r["orders"][i]) : ""
        @printf("%12.4e %12.4e %10s\n", r["dt"][i], r["err"][i], ord)
    end
end
