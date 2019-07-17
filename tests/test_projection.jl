"""This module implements the projection algorithm."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT

using StatsBase

# Project on one function
function proj_pi(f::Function,
                    t::Array{Float64,1},
                    x::Array{Float64,1},
                    i::Int,
                    data::Array{Float64,2},
                    regressionModel::RegressionModel)
    if f(x) <= 0
        return x
    else
        step = f(x)*t/(t'*t)
        if isnan(sum(step))
            if i > size(data)[1]
                x[1] = max(0,x[1])
                return x
            else
                step = equivalent(x, i, data, regressionModel)
            end
        end
        return x - step
    end
end


# Aggregator of projection
function QR(x0::Array{Float64,1},
             xn::Array{Float64,1},
             zn::Array{Float64,1})

    π = (x0-xn)'*(xn-zn)
    μ =(x0-xn)'*(x0-xn)
    ν = (xn-zn)'*(xn-zn)
    ρ = μ*ν-π^2
    if ((ρ==0.0) && (π >=0))
        return zn
    end
    if (ρ>0)
         if (π*ν>=ρ)
            return x0+(1+π/ν)*(zn-xn)
         else
            return xn+ν/ρ*(π*(x0-xn)+μ*(zn-xn))
         end
    end
    if ρ <= 0.0
        return zn
    end
    if isnan(ρ)
        # println("Warning, rho = NaN")
        # println("pi = ", π)
        # println("nu = ", ν)
        # println("mu = ", μ)
        # println("zn = ", zn)
        # println("x0 = ", x0)
        # println("xn = ", xn)
        if (π*ν>=ρ)
           return x0+(1+π/ν)*(zn-xn)
        else
           return xn+ν/ρ*(π*(x0-xn)+μ*(zn-xn))
        end
    end
    println("rho = ", ρ)
    println("error in QR")
    return nothing
end

function typeof_constraint(i::Int, robustModel::RobustModel)
    if i == length(robustModel.I0)+1
        return PositiveConstraint()
    else
        return robustModel.name
    end
end


#  Test if a point x is inside a set of constraint
function x_in_inter(x,
                     data,
                     robustModel,
                     I,
                     para::ParallelParam)

    error("'error_loss' not defined for $(typedof(s))")
end

function x_in_inter(x::Array{Float64,1},
                     data::Array{Float64,2},
                     robustModel::RobustModel,
                     I::Array{Int64,1},
                     para::Sequential)

    aux::Float64 = 0.0
    for k in 1:length(I)
        i::Int = I[k]
        name::GeneralConstraint = typeof_constraint(i, robustModel)
        aux = max(aux,fconstraint(x, data, robustModel.regressionModel, i, name))
    end
    isnan(aux) && (aux = 1.0)
    return aux
end


function x_in_inter(x::Array{Float64,1},
                     data::Array{Float64,2},
                     robustModel::RobustModel,
                     I::Array{Int64,1}, para::Parallel)
    aux = @parallel (max) for k in 1:length(I)
        max(0,fconstraint(x, data, robustModel.regressionModel, I[k], typeof_constraint(I[k], robustModel)))
    end
    isnan(aux) && (aux = 1.0)
    return aux
end

# Projection onto a set of constraint
function proj_In(xn::Array{Float64,1},
                  data::Array{Float64,2},
                  robustModel::RobustModel,
                  I::Array{Int64,1},
                  ω::Array{Float64,1},
                  para::Sequential)

    n = length(I)
    pin::Array{Float64,1} = zeros(length(xn))
    sum_op = zeros(length(xn))
    Ln = 0
    for k in 1:n
        i = I[k]
        name = typeof_constraint(i, robustModel)
        subg = subgradient(xn, data, robustModel.regressionModel, i, name)
        pin = proj_pi(x->fconstraint(x, data, robustModel.regressionModel, i, name), subg, xn, i, data, robustModel.regressionModel)
        sum_op = sum_op + ω[k]*pin
        Ln = Ln + ω[k]*((pin-xn)'*(pin-xn))
    end
    un = xn-sum_op
    Ln = Ln/(un'*un)
    return un, Ln
end


# auxilaray function to speed uo proj parallel
function auxf(i::Int,
                xn::Array{Float64,1},
                data::Array{Float64,2},
                robustModel::RobustModel,
                ω::Float64)

    name = typeof_constraint(i, robustModel)
    subg = subgradient(xn, data, robustModel.regressionModel, i, name)
    pin::Array{Float64,1} = proj_pi(x->fconstraint(x, data, robustModel.regressionModel, i, name), subg, xn, i, data, robustModel.regressionModel)
    aux::Array{Float64,1} =  ω*vcat(pin, dot((pin-xn),(pin-xn)))

    return aux
end

function proj_In(xn::Array{Float64,1},
                  data::Array{Float64,2},
                  robustModel::RobustModel,
                  I::Array{Int64,1},
                  ω::Array{Float64,1},
                  para::Parallel)

    n::Int = length(I)
    ul::Array{Float64,1} = @parallel (+) for k in 1:n
        auxf(I[k], xn, data, robustModel, ω[k])
    end
    un::Array{Float64,1} = xn - ul[1:end-1]
    Ln::Float64 = ul[end]
    Ln = Ln/dot(un, un)
    return un, Ln
end





# Projection algorithm
function algo_proj(x0::Array{Float64,1},
                    data::Array{Float64,2},
                    robustModel::RobustModel,
                    projParams::ProjParams)

    dist_mem::Float64 = projParams.precision+1
    xn::Array{Float64,1} = copy(x0)
    xn1::Array{Float64,1} = copy(xn)+ones(length(xn))
    iter = 0
    while (iter < Int(round(length(robustModel.I0)/projParams.sample))+1) || ((dist_mem > projParams.precision) && (iter < projParams.ITER_MAX))
        # println("iter = ", iter, " dm = ", dist_mem)
        iter = iter +1
        I = StatsBase.sample(robustModel.I0, projParams.sample, replace=false, ordered=true)
        push!(I, length(robustModel.I0)+1)
        ω = ones(length(I))/(length(I))
        un, Ln = proj_In(xn, data, robustModel, I, ω, projParams.para_proj)
        dist_mem = x_in_inter(xn, data, robustModel, I, projParams.para_inter)
        if (dist_mem <= 0.0) || (dot(un,un) == 0)
            Ln = 1
            # dist_mem = (xn1-xn)'*(xn1-xn)
            I = StatsBase.sample(robustModel.I0, projParams.sample, replace=false, ordered=true)
            dist_mem = x_in_inter(xn, data, robustModel, I, projParams.para_inter)
        end
        zn = xn - Ln*un
        xn1 = xn
        xn = QR(x0,xn,zn)
    end
    return xn, dist_mem, iter
end



function aux_proj(i::Int,
                    xn::Array{Float64,1},
                    data::Array{Float64,2},
                    robustModel::RobustModel,
                    ω::Float64)
    name = typeof_constraint(i, robustModel)
    subg::Array{Float64,1} = subgradient(xn, data, robustModel.regressionModel, i, name)
    pin::Array{Float64,1} = proj_pi(x->fconstraint(x, data, robustModel.regressionModel, i, name), subg, xn, i, data, robustModel.regressionModel)
    aux::Float64 = max(0, fconstraint(xn, data, robustModel.regressionModel, i, name))
    return vcat(ω*pin, ω*dot((pin-xn),(pin-xn)), aux)
end



function algo_proj_bis(x0::Array{Float64,1},
                    data::Array{Float64,2},
                    robustModel::RobustModel,
                    projParams::ProjParams)

    dist_mem::Float64 = projParams.precision+1
    xn::Array{Float64,1} = copy(x0)
    iter::Int = 0
    while (iter < Int(round(length(robustModel.I0)/projParams.sample))+1) || ((dist_mem > projParams.precision) && (iter < projParams.ITER_MAX))
        # println("iter = ", iter, " dm = ", dist_mem)
        iter = iter +1
        I = StatsBase.sample(robustModel.I0, projParams.sample, replace=false, ordered=true)
        push!(I, length(robustModel.I0)+1)
        ω = ones(length(I))/(length(I))
        res::Array{Float64,1} = @parallel (+) for k in 1:length(I)
            aux_proj(I[k], xn, data, robustModel, ω[k])
        end
        dist_mem = res[end]/length(I)
        un::Array{Float64,1} = xn-res[1:end-2]
        Ln::Float64 = res[end-1]/(un'*un)
        if (dist_mem <= 0.0) || (dot(un,un) == 0)
            Ln = 1
            # dist_mem = (xn1-xn)'*(xn1-xn)
            I = StatsBase.sample(robustModel.I0, projParams.sample, replace=false, ordered=true)
            dist_mem = x_in_inter(xn, data, robustModel, I, projParams.para_inter)
        end
        xn = QR(x0,xn,xn - Ln*un)
    end
    return xn, dist_mem, iter
end




function run_algo_bis(xn::Array{Float64,1},
                    data::Array{Float64,2},
                    robustModel::RobustModel,
                    optParams::OptParams,
                    projParams::ProjParams)

    # dist_mem = optParams.stability+1
    dm::Array{Float64,1} = []
    mini::Float64 = Inf
    mem::Array{Float64,1} = zeros(size(xn))
    y::Array{Float64,1} = zeros(size(xn))
    xn1::Array{Float64,1} = copy(xn)
    iter::Int = 0
    while (iter < optParams.itmax)
        iter = iter +1
        #print(iter," ")
        # xn = xn1 - optParams.learning_rate*robustModel.descent_direction
        y = xn + ((iter-2)/(iter+1))*(xn - xn1)
        xn, DM, it = algo_proj_bis(y-optParams.learning_rate*robustModel.descent_direction,
                                data,
                                robustModel,
                                projParams)
        (robustModel.descent_direction'*xn < mini) && (mini = min(mini,robustModel.descent_direction'*xn); mem = xn )
        push!(dm, robustModel.descent_direction'*xn)
        xn1 = xn
        ((iter % optParams.verbosity) == 0) && print(iter," ",DM," ")
    end
    # println("Exit after ", iter, " iterations")
    return xn, y, dm, mem, mini
end








# optimization algorithm
function run_algo(xn::Array{Float64,1},
                    data::Array{Float64,2},
                    robustModel::RobustModel,
                    optParams::OptParams,
                    projParams::ProjParams)

    # dist_mem = optParams.stability+1
    dm::Array{Float64,1} = []
    t_iter::Array{Float64,1} = []
    mini::Float64 = Inf
    mem::Array{Float64,1} = zeros(size(xn))
    y::Array{Float64,1} = zeros(size(xn))
    xn1::Array{Float64,1} = copy(xn)
    iter::Int = 0
    # while (iter < optParams.itmax) || (iter > 2 &&  (dm[end]-dm[end-1]<optParams.stability))
    while (iter < optParams.itmax)
        iter = iter +1
        #print(iter," ")
        # xn = xn1 - optParams.learning_rate*robustModel.descent_direction
        y = xn + ((iter-2)/(iter+1))*(xn - xn1)
        tic()
        xn, DM, it = algo_proj(y-optParams.learning_rate*robustModel.descent_direction,
                                data,
                                robustModel,
                                projParams)
        time_iter = toq()
        (robustModel.descent_direction'*xn < mini) && (mini = min(mini,robustModel.descent_direction'*xn); mem = xn )
        push!(dm, robustModel.descent_direction'*xn)
        push!(t_iter, time_iter)
        xn1 = xn
        ((iter % optParams.verbosity) == 0) && print(iter," ",DM," ")
    end
    # println("Exit after ", iter, " iterations")
    return xn, y, dm, mem, mini, t_iter
end
