"""This module implements classical constraints and their gradients."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT

function div_function(nameDiv::DivergenceConstraint)
    error("'error_loss' not defined for $(typedof(s))")
end

function div_function(nameDiv::KLConstraint)
    return (f = x->exp(x)-1)
end




function div_grad(nameDiv::DivergenceConstraint)
    error("'error_loss' not defined for $(typedof(s))")
end

function div_grad(nameDiv::KLConstraint)
    return (f = x->exp(x))
end




function fconstraint(x, Z, regressionModel, i, constraint::GeneralConstraint)
    error("'error_loss' not defined for $(typedof(s))")
end

function fconstraint(x, Z, regressionModel, i, constraint::PositiveConstraint)
    return -x[1]
end

function fconstraint(x, Z, regressionModel, i, constraint::DivergenceConstraint)
    lt = size(Z)[2]-1

    λ = x[1]
    μ = x[2]
    θ = x[3:3+lt-1]
    s = x[3+lt:end]

    # f = x->exp(x)-1
    # f = constraint.f
    f = div_function(constraint)

    # inside_number = (error_loss(θ, Z[i,:], regressionModel) - μ)/λ

    if (λ<=0)
        return error_loss(θ, Z[i,:], regressionModel)-μ
    else
        return λ*f((error_loss(θ, Z[i,:], regressionModel)-μ)/λ)-s[i]
    end
end

function fconstraint(x, Z, regressionModel, i, constraint::EntropicConstraint)
    lt = size(Z)[2]-1

    μ = x[1]
    θ = x[2:2+lt-1]
    s = x[2+lt:end]

    # f = x->exp(x)-1
    # f = constraint.f
    f = div_function(KLConstraint())

    return f((error_loss(θ, Z[i,:], regressionModel)-μ))-s[i]

end


function fconstraint(x, Z, regressionModel, i, constraint::DROConstraint)
    lt = size(Z)[2]-1

    τ = x[1]
    θ = x[2:2+lt-1]
    s = x[2+lt:end]

    N = length(s)

    l = div(i-1,N)+1
    j = i-(l-1)*N

    return error_loss(θ, Z[j,:], regressionModel)- τ*(Z[l,:]-Z[j,:])'*(Z[l,:]-Z[j,:])-s[l]
end



# function circle(x,i)
#      return (x[1]-Z[i,1])^2 + (x[2]-Z[i,2])^2 - Z[i,3]^2
# end
#
# function expo(x,i)
#     return exp(Z[i,1]*x[1]) - x[2]
# end
# if fname == "circle"
#          return [2*(x[1]-Z[i,1]); 2*(x[2]-Z[i,2])]
#     elseif fname == "expo"
#         return [Z[i,1]*exp(Z[i,1]*x[1]); -1]



#
# function subgradient_lambda(f, fp, λ, inside_number, loss_eval)
#     return f(inside_number) - loss_eval/λ*fp(inside_number)
# end
#
# function subgradient_mu(fp, inside_number)
#     return -fp(inside_number)
# end
#
# function subgradient_theta(fp, λ, inside_number, lossp)
#     grad = zeros(length(lossp))
#     for k in 1:length(lossp)
#         grad[k] = lossp[k]*fp(inside_number)
#     end
#     return grad
# end



function subgradient(x, Z, regressionModel, i, constraint::GeneralConstraint)
    error("'error_loss' not defined for $(typedof(s))")
end

function subgradient(x, Z, regressionModel, i, constraint::PositiveConstraint)
    return [-1; zeros(length(x)-1)]
end


function subgradient(x, Z, regressionModel, i, constraint::DivergenceConstraint)
    lt = size(Z)[2]-1

    λ = x[1]
    μ = x[2]
    θ = x[3:3+lt-1]
    s = x[3+lt:end]

    # f  = x->exp(x)-1
    # fp = x->exp(x)
    f  = div_function(constraint)
    fp = div_grad(constraint)

    inside_number = (error_loss(θ, Z[i,:], regressionModel) - μ)/λ
    lossp = zeros(lt)
    for k in 1:lt
        lossp[k] = error_prime(θ, Z[i,:], k, regressionModel)
    end

    if  λ<=0

        grad = zeros(length(x))
        grad[2] = -1
        grad[3:3+lt-1] = lossp
        grad[3+lt:end] = zeros(length(s))

        return grad
    else

        grad = zeros(length(x))
        # grad[1] = subgradient_lambda(f, fp, λ, inside_number, (error_loss(θ, Z[i,:], regressionModel) - μ))
        # grad[2] = subgradient_mu(fp, inside_number)
        # grad[3:3+lt-1] = subgradient_theta(fp, λ, inside_number, lossp)
        grad[1] = f(inside_number) - (error_loss(θ, Z[i,:], regressionModel) - μ)/λ*fp(inside_number)
        grad[2] = -fp(inside_number)
        for k in 1:length(lossp)
            grad[3+k-1] = lossp[k]*fp(inside_number)
        end
        grad[3+lt-1+i] = -1
        return grad
    end
end


function subgradient(x, Z, regressionModel, i, constraint::EntropicConstraint)
    lt = size(Z)[2]-1

    μ = x[1]
    θ = x[2:2+lt-1]
    s = x[2+lt:end]

    # f  = x->exp(x)-1
    # fp = x->exp(x)
    f  = div_function(KLConstraint())
    fp = div_grad(KLConstraint())

    inside_number = (error_loss(θ, Z[i,:], regressionModel) - μ)
    lossp = zeros(lt)
    for k in 1:lt
        lossp[k] = error_prime(θ, Z[i,:], k, regressionModel)
    end

    grad = zeros(length(x))
    grad[1] = -fp(inside_number)
    for k in 1:length(lossp)
        grad[2+k-1] = lossp[k]*fp(inside_number)
    end
    grad[2+lt-1+i] = -1
    return grad

end




function subgradient(x, Z, regressionModel, i, constraint::DROConstraint)
    lt = size(Z)[2]-1

    τ = x[1]
    θ = x[2:2+lt-1]
    s = x[2+lt:end]

    N = length(s)

    l = div(i-1,N)+1
    j = i-(l-1)*N

    lossp = zeros(lt)
    for k in 1:lt
        lossp[k] = error_prime(θ, Z[j,:], k, regressionModel)
    end

    grad = zeros(length(x))
    grad[1] = -(Z[l,:]-Z[j,:])'*(Z[l,:]-Z[j,:])
    grad[2:2+lt-1] = lossp
    grad[2+lt-1+l] = -1

    return grad
end
