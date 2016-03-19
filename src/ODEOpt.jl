using Optim
module ODEOpt
using  ODE
# package code goes here
export Solution, Record, fgen, g, grav, extract, solutionplot, soserr, mid, obj, tprint, solve45,
       secondorder, solution, ivp 

#define ODEs
immutable Solution{T}
    time::Vector{T}
    y::Vector{Vector{T}}
end

""" fgen(β): return a Function describing ODE of oscillator.
    second order homogeneous system for 1d vibration
    x'' + x' = 0
    x' = v
    v' = -x
"""
function fgen(β)
    function f(t,y)
        return [y[2]; -β*y[1]]
    end
    return f
end
f = fgen(1.0)


""" g(t,y)
second order inhomogeneous system for 1d vibration
x'' + x' = f(t)
Let x' = v
v' + v = f(t)
v' = f(t) - v
"""
function g(t, y)
    #f(t) = t
    α=-1
    β=+2
    (x,v) = y
    return [v;α*t-β*x]
end

β=1
function grav(t, y)
    #f(t) = t
    (x,v) = y
    return [v;β*min(-x,0)]::Vector{Float64}
end

function grav(β)
    #f(t) = t
    function f(t,y)
        (x,v) = y
        return [v;β*min(-x,0)]::Vector{Float64}
    end
    return f
end

"""extract(y): get the position (x) and velocity (v) from the system state (y)"""
function extract(y)
    x = Float64[p[1] for p in y]
    v = Float64[p[2] for p in y]
    return x,v
end
function extract(sol::Solution)
    y = sol.y
    x = Float64[p[1] for p in y]
    v = Float64[p[2] for p in y]
    return sol.time, x, v
end

"""mid(x,y)
midpoint of x and y
"""
function mid(x,y)
    return (x+y)/2
end

# TODO: use Interpolations.jl for this
function mid(t, tx, x, ty, y)
    α = (t-tx)/(ty-tx)
    return α*x +(1-α)*y
end

"""soserr{T}(timex::Vector{T}, x::Vector{T}, timedata::Vector{T}, data::Vector{T})

Sum of squared error between x and data, timex is the domain of the function x
and timedata is the domain of function data. The iterpolation should be done better
using a package such as Interpolations.jl.
"""
function soserr{T}(timex::Vector{T}, x::Vector{T}, timedata::Vector{T}, data::Vector{T})
    # TODO: This implementation is inefficient O(log(n)*n) instead of O(n) because
    # the interpolation is done with binary search instead of an actual interpolation
    # algorithm.
    sos = 0.0
    L = length(timex)
    for (i,t) in enumerate(timedata)
        # TODO: use Interpolations.jl for this
        j = searchsorted(timex, t).start
        j = max(1, j)
        jn = min(j+1, L)
        solution = mid(x[j],x[jn])
        #= solution = mid(t, timex[j], x[j], timex[jn], x[jn]) =#
        sos += (solution-data[i])^2
    end
    return sos
end

"""obj(f::Function, time::Vector{Float64}, data)

Solve the ode described by f on the domain time and compare to data
which was measured on the domain time.

- f is passed to a Runge-Kutta method from ODE.jl
- the solution is interpolate as described in soserr
- the result is the sum of squared errors.
"""
function obj(f::Function, ystart::Vector{Float64}, time::Vector{Float64}, data)
    tsolve, y = ode45(f, ystart, time, abstol=1e-16, reltol=1e-13, points=:specified)
    x, v = extract(y)
    #= @show norm(tsolve .- time) =#
    d² = soserr(tsolve, x, time, data)
    return tsolve, x, v, d²
end

function solve45(f::Function, ystart::Vector{Float64}, time)
    return Solution(ode45(f, ystart, time)...)
end

include("secondorder.jl")

using PyPlot

"""solutionplot(time::Vector, x::Vector, v::Vector)
plot the solution as position and velocity over the time domain.
"""
function solutionplot(time::Vector, x::Vector, v::Vector)
    plot(time, x, label="x")
    plot(time, v, label="v")
    legend()
end
function solutionplot(sol::Solution)
    return solutionplot(extract(sol)...)
end

type Record{T, R}
    β::T
    ystart::Vector{Float64}
    nsamples::Int
    res::R
end

function measure(r::Record)
    β⁺ = r.res.minimum
    err = r.β-β⁺
    fval = r.res.f_minimum
    return β⁺, err, fval
end

function tprint(io, r::Record)
    β=r.β
    β⁺, err, fval = measure(r)
    k = r.nsamples
    println(io, "$k\t$β\t$β⁺\t$err\t$fval")
end

end # module

