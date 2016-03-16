using ODEOpt
using Base.Test

# write your own tests here
@test 1 == 1

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
    #= solutionplot(time, x,v) =#
    tol = 1e-2
    @test_approx_eq_eps  time[0.0 .<= abs(x).<=tol][end] 2.9458917835671343 1e-8
end
testgrav()
