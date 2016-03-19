using Optim
using ODEOpt
using ODE
using PyPlot
using Base.Test
using NLsolve
using ForwardDiff
PLOT = true
let
    ystart = [0.0;1.0]
    trange = linspace(0,2π,100)
    a, b, c = 1.0, 2.0, 3.0
    foo = secondorder(a,b,c)
    @test foo(0,[0;1]) == [1.0;-2.0]
end

let
    # Test based on the example at
    # http://www.sosmath.com/diffeq/second/constantcof/constantcof.html
    #
    #     x'' + 2x' + 2x = 0
    #     x(π/4) = 2; x'(π/4) = -2
    #
    #     convert to first order using the system
    #     y[1] = x; y[2] = x'
    #
    #     descretize at linspace(π/4,2π,100)
    #
    # solve the ode numerically then plug into the formulas to
    # check that the formulas match the ode45 solution.
    ystart = [2.0;-2.0]
    trange = linspace(π/4,2π,100)
    a, b, c = 1.0, 2.0, 2.0
    foo  = secondorder(a,b,c)
    sol = solve45(foo, ystart, collect(trange))
    result, formula, deriv = ivp(a,b,c, ystart, π/4)
    @show params = result.zero
    @show result.trace
    @show result
    # figure()
    # test that the computed solution matches the formula
    form = [formula(t,params...) for t in sol.time]
    deri = [deriv(t,params...) for t in sol.time]
    @test norm(form-[y[1] for y in sol.y]) <= 1e-3
    @test norm(deri-[y[2] for y in sol.y]) <= 1e-3
    # plot general solution.
    if PLOT
        plot(sol.time, form, label="formula")
        plot(sol.time, deri, label="f'")
        title("parameter fitting \$ $a \\ddot{x} + $b \\dot{x} + $c x = 0\$") 
        solutionplot(sol)
    end
end

let
    # parameter fitting for 2nd order linear ode
    # the equation is ax'' + bx' + cx = 0
    # we want to recover the coefficients a,b,c as well as the initial values x₀,v₀
    # generate "exact" solution
    info("Testing Parameter Fitting")
    ystart = [2.0;-2.0]
    trange = linspace(π/4,2π,100)
    a, b, c = 1.0, 2.0, 2.0
    params = [a,b,c]
    # initial parameter guess must be fairly close to avoid local optima
    param₀ = 1.5*ones(3)
    #param₀ = params .+ randn(3)/5

    foo  = secondorder(a,b,c)
    sol = solve45(foo, ystart, collect(trange))
    result, formula, deriv = ivp(a,b,c, ystart, π/4)
    consts = result.zero
    form = Float64[formula(t,consts...) for t in sol.time]
    deri = Float64[deriv(t,consts...) for t in sol.time]
    
    # add noise to get data as if measured
    srand(1)
    noisev = 0.10 * randn(length(form))
    data = form .+ noisev
    # x = [y[1] for y in sol.y] 
    # make sure the soserror matches the norm of the measurement noise.
    # @show sqrt(soserr(sol.time,form, sol.time, data)) norm(noisev) 1e-2
    tsolve_opt, x⁺_opt, v_opt, d²_opt = obj(foo, ystart, sol.time, data)
    @test_approx_eq_eps sqrt(d²_opt) norm(noisev) 1e-2

    # fit the parameters to the data 
    @show param₀
    γ(x) = sqrt(obj(secondorder(x...), ystart, sol.time, data)[end])
    @show res = optimize(γ,
                         param₀,
                         xtol=1e-12,
                         ftol=1e-32,
                         method=ConjugateGradient(),
                         #method=GradientDescent(),
                         #method=LBFGS(),
                         autodiff=false,
                         show_trace=true,
                         iterations=1000,
                         )
    # report the quality of the fit
    @show param⁼ = res.minimum
    @show res.f_minimum d²_opt
    tsolve, x⁺, v⁺, d² = obj(secondorder(param⁼...), ystart, sol.time, data)
    if PLOT
        figure()
        plot(tsolve, x⁺, label="\$x^+_o\$")
        plot(sol.time,form,label="\$x_f\$")
        plot(tsolve, x⁺, label="\$x^+\$")
        plot(sol.time, data, ".",  label="data")
        title("parameter fitting \$ $a \\ddot{x} + $b \\dot{x} + $c x = 0\$") 
        legend()
    end
    # check that residual is small
    @show sqrt(soserr(tsolve, x⁺, sol.time, form))
    @test_approx_eq_eps γ(res.minimum) sqrt(d²_opt) 2e-1
    #check that error is small
    @show err = norm(param⁼ - params)
    @test err < 3e-1
end
