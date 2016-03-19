# src/secondorder.jl contains function for solving second order ODEs
# can use either numerical integration or closed for formulas
# secondorder(a,b,c) generates the function you pass to ODE.jl RK methods
# solution(a,b,c) is the closed form solver
# ivp solves the initial value problem.
using Optim
using NLsolve

""" secondorder(α,β,γ) yields a function f
    which calculates the ode rule

        αx'' + βx' + γx = 0

    using the substitution

        y = [x; x']

        y[1]' = y[2]

        y[2]' = -α^-1 (βy[2] + γy[1])

"""
function secondorder(α, β, γ)
        return (t, y) -> [y[2]; -(β*y[2] + γ*y[1])/α]
end

function secondorder(x::Vector)
    return secondorder(x...)
end

"""descriminant(a,b,c): quadtratic equation descriminant of ax^2 + bx + c = 0
"""
descriminant(a,b,c) = √(complex(b^2-4a*c))

"""roots(a,b,c): solve the quadratic equation ax^2 + bx + c = 0 """
function roots(a,b,c) 
    d = descriminant(a,b,c);
    return (-b+d)/2a, (-b-d)/2a
end 

doc"""solution(a,b,c,y₀,t₀) general solution of second order linear homogenous

Arguments:
   - a,b,c::Real the coeffients of the ODE

Returns:
   - func(t, c₁, c₂): the solution at t when the coefficients are c₁, c₂
   - deriv(t, c₁, c₂): the derivative at t when the coefficients are c₁, c₂

Notes: func is derived from the textbook definition and deriv is based on
 applying calculus rules to that (a symbolic algebra tool did this).
"""
function solution(a,b,c)
    @show d = descriminant(a,b,c)
    @show r₁, r₂ = roots(a,b,c)
    if real(d) == d
        info("characteristic polynomial has real roots")
        return (t, c₁, c₂) -> c₁ * exp(r₁*t) + c₂*exp(r₂*t),
               (t, c₁, c₂) -> c₁*r₁*exp(r₁*t) + c₂*r₂*exp(r₂*t)
               
    end
    if imag(d) != 0.0
        info("characteristic polynomial has complex roots")
        @show α, β = -b/2a, imag(d/2a)
        func(t, c₁, c) = begin
            return (c₁*exp(α*t)*cos(β*t) + c*exp(α*t)*sin(β*t))
        end
        deriv(t,c₁,c₂) = begin
            expα = exp(α*t)
            cosβ = cos(β*t)
            sinβ = sin(β*t)
            return (-β*c₁*expα*sinβ +
                    α*c₂*expα*sinβ +
                    α*c₁*expα*cosβ +
                    β*c₂*expα*cosβ
                    )
            end
        return func, deriv
    end
end

"""ivp(a,b,c,y₀,t₀) solves second order linear homogenous initial value problem

Arguments:
   - a,b,c::Real the coeffients of the ODE
   - y₀ the initial value as a vector
   - t₀ the initial time value

Returns:
   - result : the Optim.jl result object of solving the equations
   - general: the general form solution (see `solution`)
   - deriv:   the general form derivative (see `solution`)

Note: result.zero is the coefficients to plug into the general solution 
to get the correct particular solution.
"""
function ivp(a,b,c, y₀, t₀)
    x₀ = y₀[1]
    v₀ = y₀[2]
    general, deriv = solution(a,b,c)
    
    function f!(params, fvec)
        fvec[1] = x₀ - general(t₀, params...)
        fvec[2] = v₀ - deriv(t₀, params...)
    end
    result = nlsolve(f!,
                     [0.0,0.0],
                     autodiff=true,
                     method=:trust_region,
                     store_trace=true)
    return result, general, deriv
end
