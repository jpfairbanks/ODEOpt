using ODEOpt
using ODE
using Base.Test

function testgrav()
    # create time samples
    tmin = 0.0
    tmax = 10
    nsamples = 500
    trange = linspace(tmin, tmax, nsamples)
    #initial position
    ystart = [0.2; 1.0]::Vector{Float64}
    time, y = ode45(grav, ystart, trange)
    x,v = extract(y)
    tol = 1e-2
    @test_approx_eq_eps  time[0.0 .<= abs(x).<=tol][end] 2.9458917835671343 1e-8
end
testgrav()

function noise(σ, n::Integer)
    r = randn(n)
    return σ*r
end

using Optim
using PyPlot
let
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

σ = 0.01
PLOT = true
samplefactor = 2


function simulate(fgen::Function, β::Float64, sol::Solution, samples, PLOT::Bool)
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
    return rec
end

recs = Record[]
f = fgen(β)
sol = Solution(ode45(f, ystart, trange)...)
function test_smallerr(recs)
    @test_approx_eq_eps recs[end].β recs[end].res.minimum 1e-1
end

for samplefactor in [1,2,3,4,5,6,7,8,8,9,10]
    samples = 1:samplefactor:length(trange)
    #= @show samples length(trange) length(samples) length(trange[samples]) =#
    r = simulate(fgen, β, sol, samples, false)
    push!(recs, r)
    test_smallerr(recs)
end

println(STDOUT, "n\tβ\tβ⁺\terr\tfval")
for (i,r) in enumerate(recs)
    tprint(STDOUT, r)
end

samplefactor=1
for β in float([1,2,3,4,5,6,7,8,8,9,10])
    βmax = 3β
    f = fgen(β)
    sol = Solution(ode45(f, ystart, trange)...)
    samples = 1:samplefactor:length(trange)
    r = simulate(fgen, β, sol, samples, β==10?true:false)
    push!(recs, r)
    test_smallerr(recs)
end
println(STDOUT, "n\tβ\tβ⁺\terr\tfval")
for (i,r) in enumerate(recs)
    tprint(STDOUT, r)
end
end
include("secondorder.jl")

