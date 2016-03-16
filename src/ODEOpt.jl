using Optim
module ODEOpt
using  ODE
# package code goes here
export Solution, Record, fgen, g, grav, extract, solutionplot, soserr, mid, obj, tprint

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
        return [y[2]; -β*y[1]]::Vector{Float64}
    end
    return f
end
f = fgen(1.0)

# second order inhomogeneous system for 1d vibration
# x'' + x' = f(t)
# Let x' = v
# v' + v = f(t)
# v' = f(t) - v
α=-1
β=+2
function g(t, y)
    #f(t) = t
    (x,v) = y
    return [v;α*t-β*x]::Vector{Float64}
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
    tsolve, y = ode45(f, ystart, time)
    x, v = extract(y)
    #= @show norm(tsolve .- time) =#
    d² = soserr(tsolve, x, time, data)
    return tsolve, x, v, d²
end

using PyPlot

"""solutionplot(time::Vector, x::Vector, v::Vector)
plot the solution as position and velocity over the time domain.
"""
function solutionplot(time::Vector, x::Vector, v::Vector)
    plot(time, x, label="x")
    plot(time, v, label="v")
    legend()
end

type Record{T}
    β::T
    ystart::Vector{Float64}
    nsamples::Int
    res
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

function noise(σ, n::Integer)
    r = randn(n)
    return σ*r
end

using Optim
using ODE
using ODEOpt
using PyPlot
using Base.Test

# create time samples
srand(0)
tmin = 0.0
tmax = 10
ntimes = 400
trange = linspace(tmin, tmax, ntimes)
#initial position
ystart = [0.0; 1.0]::Vector{Float64}
β = 1.0
βmin = 0.0
βmax = 4.0

σ = 0.1
PLOT = true
samplefactor = 2




function simulate(β::Float64, sol::Solution, samples, PLOT::Bool)
    t, x,v = extract(sol)
    r = noise(σ, length(samples))
    data = x[samples] .+  r
    tdata = t[samples]
    #= @show tdata[end] t[end] length(t) length(tdata) =#
    # TODO: why are the samples missing the end of the domain ?
    #       the samples IntRange is constructed from a different solution from sol.
    # TODO: sample uniformly rather than at Runge Kutta points.
    if PLOT
        solutionplot(t, x,v)
        plot(tdata, data, "o", label="data")
        legend()
    end

    @show norm(σ*r)
    z = soserr(t, x, tdata, data,)
    #= @test_approx_eq_eps norm(σ*r) z 1e-8 =#

    @show sqrt(soserr(t, x, tdata, data))
    tsolve, xsolve, vsolve, d² = obj(f, ystart, tdata, data)
    @show sqrt(d²)

    γ(x) = obj(fgen(x), ystart, tdata, data)[end]
    @show res = optimize(γ, βmin, βmax)

    β⁺ = res.minimum
    t⁺, y⁺ = ode45(fgen(β⁺), ystart, trange)
    x⁺, v⁺ = extract(y⁺)
    if PLOT
        solutionplot(t⁺, x⁺, v⁺)
    end

    rec =  Record(β, ystart, length(samples),  res)
    #= ODEOpt.tprint(STDOUT, rec) =#
    return rec
end

recs = Record[]
f = fgen(β)
sol = Solution(ode45(f, ystart, trange)...)
for samplefactor in [1,2,3,4,5,6,7,8,8,9,10]
    samples = 1:samplefactor:length(trange)
    #= @show samples length(trange) length(samples) length(trange[samples]) =#
    r = simulate(β, sol, samples, false)
    push!(recs, r)
end

println(STDOUT, "n\tβ\tβ⁺\terr\tfval")
for (i,r) in enumerate(recs)
    tprint(STDOUT, r)
end

samplefactor=1
for β in float([1,2,3,4,5,6,7,8,8,9,10])
    βmax = 4β
    f = fgen(β)
    sol = Solution(ode45(f, ystart, trange)...)
    samples = 1:samplefactor:length(trange)
    r = simulate(β, sol, samples, β==10?true:false)
    push!(recs, r)
end
println(STDOUT, "n\tβ\tβ⁺\terr\tfval")
for (i,r) in enumerate(recs)
    tprint(STDOUT, r)
end

