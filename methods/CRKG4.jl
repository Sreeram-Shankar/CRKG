# CRKG4 — Chain Runge–Kutta–Gegenbauer: a real-arithmetic, natively 4th-order stabilized
# Runge–Kutta method. It is natively 4th-order
# for nonlinear problems, not requiring splitting or substepping or extrapolation. 
# The only other stabilized rk methods to get to these orders are ROCK and DUMKA.
# It has far better internal stability than these methods, and is specifically designed
# to be designed for limiting: it has small amplication and physical internal stages
# (each stage polynomial is the stability function of a shorter method of the same type,
# so every Y_j is a legitimate approximate solution that can be limited).
#
#
# Stability polynomial. With M "segments" there are L = 4M stages per blend component. The target
# stability polynomial is a weighted sum of Gegenbauer polynomials whose degrees are
# multiples of M, rescaled so that the negative real interval from -beta to 0 maps onto
# the interval where those polynomials stay bounded. Its weights make it match the
# exponential to third order, and beta is made as large as stability allows. The
# Gegenbauer parameter nu and the damping shift (damp) trade a little interval length
# for damping within the stability interval. Because gegenbauer polynomials are not very smooth,
# these damping additions make the polynomial smoother and reduce oscilations. 
# For some reason I find that nu = 0.5 (Legendre polynomials) works best, with damp = DAMP = 0.5 and the damping level ETA = 0.8.
# It produces good smoothing without blowing up the internal amplification.
#
# Each component is a real three-term recurrence,
# The components differ in their Gegenbauer parameter nu, their damping, and their coverage factor M.
# and damping, so they behave differently on nonlinear problems. Every intermediate
# state is real and physical: Y_j equals the step of the j-stage chain with the same
# Gegenbauer coefficients.
#
# Every stage of every component gets its own real weight, and the new solution
# is the starting state plus the step times the weighted sum of all stage right-hand
# sides. The weights are solved once, offline, so that the blended method has exactly
# the target stability polynomial, (so unlike CFRKG — chain-factorized RKG — where it is different ordering combinations 
# of the same polynomial factorization, this uses a different polynomial for each component), 
# satisfies all third-order conditions, and has a small
# fourth-order error. A second set of weights gives an embedded second-order pair.
# The target polynomial is the one that we make with the provided nu damp and s, 
# and the different linear combinations of varying polynomials gives the weights to form the 
# same stability region as that one but with nonlinear 3rd order and the minimum internal amplification.
# However, this by itself is 2x the amount of RHS evaluations. to get this down, 
# we find a combination of chains that reuse some stages, picked so that their maximum internal amplification is within 5x of the best one
# by default. this can be changed but we recommend keeping it at 5x, since results are diminishing after it. 
#
# Limiting. The limiter is applied to each new stage as soon as it is formed, so every
# right-hand side is evaluated on an admissible state, and to the new solution.

#this file is essentially just a wrapper the CRKG_common file but for hte 4th order method
module CRKG4

using LinearAlgebra, Printf, Serialization, Polyester

export design, min_M, workspace, step!, solve!

#sets the differnt variables for the 4th order method
const ORDER = 4
const CACHE_DIR = joinpath(@__DIR__, "crkg4_cache")
const ERR_KEY = :err5
const NCHAINS_DEFAULT = 3 #3 chains (a trunk plus two branches) are default for the 4th order method
# we could use less but internal amplication takes a hit. using more is more expensive. this is the best balance

include("CRKG_design.jl")
include("CRKG_common.jl")

end # module
