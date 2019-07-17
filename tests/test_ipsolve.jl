"""This module implements interface for robust optimization with a solver."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT

function KL_opt(Z, ϵ, solver, regression::LinearRegression)
    N = size(Z)[1]
    lt = size(Z)[2]-1
    X = Z[:,1:end-1]
    Y = Z[:,end]

    m = Model(solver=solver)

    @variable(m,θ[1:lt])
    @variable(m, λ >= 0)
    @variable(m, μ)
    @variable(m, s[1:N])

    @NLconstraint(m, cons[i = 1:N], ((sum(θ[j]*X[i,j] for j in 1:lt) - Y[i])^2-μ) <= λ*(log(s[i]+λ) - log(λ)))

    @objective(m, Min, λ*ϵ+μ + sum(1/N*s[i] for i in 1:N))

    status = solve(m)

    return getvalue(θ)
end

function DRO_opt(Z, ϵ, solver, regression::LinearRegression)
    N = size(Z)[1]
    lt = size(Z)[2]-1
    X = Z[:,1:end-1]
    Y = Z[:,end]

    m = Model(solver=solver)

    @variable(m, θ[1:lt])
    @variable(m, τ >= 0)
    @variable(m, s[1:N])

    @NLconstraint(m, myconstr[i=1:N,j=1:N], (sum(θ[k]*Z[j,k] for k in 1:lt)- Z[j,end])^2 - τ*sum((Z[i,k]-Z[j,k])*(Z[i,k]-Z[j,k]) for k in 1:lt+1) <= s[i])

    @objective(m, Min, τ*ϵ + sum(1/N*s[i] for i in 1:N))

    # print(m)

    status = solve(m)
    # println("theta = ", getvalue(θ))
    # println("tau = ", getvalue(τ))
    # println("s = ", getvalue(s))
    return getvalue(θ)
end








function KL_opt(Z, ϵ, solver, regression::LogisticRegression)
    N = size(Z)[1]
    lt = size(Z)[2]-1
    X = Z[:,1:end-1]
    Y = Z[:,end]

    m = Model(solver=solver)

    @variable(m,θ[1:lt])
    @variable(m, λ >= 0)
    @variable(m, μ)
    @variable(m, s[1:N])

    @NLconstraint(m, cons[i = 1:N], (log(1+exp(-Y[i]*sum(θ[j]*X[i,j] for j in 1:lt)))-μ) <= λ*(log(s[i]+λ) - log(λ)))



    @objective(m, Min, λ*ϵ+μ + sum(1/N*s[i] for i in 1:N))

    status = solve(m)

    return getvalue(θ)
end

function DRO_opt(Z, ϵ, solver, regression::LogisticRegression)
    N = size(Z)[1]
    lt = size(Z)[2]-1
    X = Z[:,1:end-1]
    Y = Z[:,end]

    m = Model(solver=solver)

    @variable(m, θ[1:lt])
    @variable(m, τ >= 0)
    @variable(m, s[1:N])

    @NLconstraint(m, myconstr[i=1:N,j=1:N], log(1+exp(-Y[i]*sum(θ[k]*X[i,k] for k in 1:lt))) - τ*sum((Z[i,k]-Z[j,k])*(Z[i,k]-Z[j,k]) for k in 1:lt+1) <= s[j])

    @objective(m, Min, τ*ϵ + sum(1/N*s[i] for i in 1:N))

    # print(m)

    status = solve(m)
    # println("theta = ", getvalue(θ))
    # println("tau = ", getvalue(τ))
    # println("s = ", getvalue(s))
    return getvalue(θ)
end








function normal_opt(Z, solver, regression::LinearRegression)
    N = size(Z)[1]
    lt = size(Z)[2]-1
    X = Z[:,1:end-1]
    Y = Z[:,end]

    m = Model(solver=solver)

    @variable(m,θ[1:lt])

    @NLobjective(m, Min, sum(1/N*(sum(θ[j]*X[i,j] for j in 1:lt) - Y[i])^2 for i in 1:N))

    status = solve(m)

    return getvalue(θ)
end



function normal_opt(Z, solver, regression::LogisticRegression)
    N = size(Z)[1]
    lt = size(Z)[2]-1
    X = Z[:,1:end-1]
    Y = Z[:,end]

    m = Model(solver=solver)

    @variable(m,θ[1:lt])


    @NLobjective(m, Min, sum( log(1+exp(-Y[i]*sum(θ[j]*X[i,j] for j in 1:lt))) for i in 1:N ))

    status = solve(m)

    return getvalue(θ)
end
