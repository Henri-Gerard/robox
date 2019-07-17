"""This module implements error losses, their gradients and equivalents."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT

function error_loss(θ, z, regressionModel::RegressionModel)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------
    Name of constraint
    """
    error("'error_loss' not defined for $(typedof(s))")
end

function error_loss(θ, z, linearRegression::LinearRegression)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------
    Name of constraint
    """
    return (θ'*z[1:end-1] - z[end])^2
end

function error_loss(θ, z, logisticRegression::LogisticRegression)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------
    Name of constraint
    """
    return log(1+exp(-z[end]*(θ'*z[1:end-1])))
end




function error_prime(θ, z, k, regressionModel::RegressionModel)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    error("'error_loss' not defined for $(typedof(s))")
end

function error_prime(θ, z, k, regressionModel::LinearRegression)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    return 2*z[k]*(θ'*z[1:end-1] - z[end])
end

function error_prime(θ, z, k, regressionModel::LogisticRegression)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    return -(z[end]*z[k])/(1+exp(z[end]*(θ'*z[1:end-1])))
end





function equivalent(x, i, Z, regressionModel::RegressionModel)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    error("'error_loss' not defined for $(typedof(s))")
end

function equivalent(x, i, Z, regressionModel::LinearRegression)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    lt = size(Z)[2]-1

    λ = x[1]
    μ = x[2]
    θ = x[3:3+lt-1]
    s = x[3+lt:end]
    normZ = Z[i,1:end-1]'*Z[i,1:end-1]
    gthemu = (error_loss(θ,Z[i,:],regressionModel)-μ)^2
    step = zeros(length(x))
    step[1] = -λ^2(error_loss(θ,Z[i,:],regressionModel)-μ)/(4*normZ*error_loss(θ,Z[i,:],regressionModel)+gthemu-1)
    step[2] = -λ^2/(4*normZ*error_loss(θ,Z[i,:],regressionModel)+gthemu-1)
    for k in 1:lt-1
        step[3-1+k] = 2*λ^2*Z[i,k]*(θ'*Z[i,1:end-1] - Z[i,end])/(4*normZ*error_loss(θ,Z[i,:],regressionModel)+gthemu-1)
    end
    step[3+lt-1+i] = -λ^3/(4*normZ*error_loss(θ,Z[i,:],regressionModel)+gthemu-1)

    return step
end

function equivalent(x, i, Z, regressionModel::LogisticRegression)
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    lt = size(Z)[2]-1

    λ = x[1]
    μ = x[2]
    θ = x[3:3+lt-1]
    s = x[3+lt:end]
    cik = (Z[i,end]^2)/(1+exp(-Z[i,end]*θ'Z[i,1:end-1]))^2*Z[i,1:end-1]'*Z[i,1:end-1]
    gthemu = (error_loss(θ,Z[i,:],regressionModel)-μ)^2
    step = zeros(length(x))
    step[1] = -λ^2(error_loss(θ,Z[i,:],regressionModel)-μ)/(cik+gthemu-1)
    step[2] = -λ^2/(cik+gthemu-1)
    for k in 1:lt-1
        step[3-1+k] = -λ^2/(cik+gthemu-1)*(Z[i,end]*Z[i,k])/(exp(error_loss(θ,Z[i,:],regressionModel)))
    end
    step[3+lt-1+i] = -λ^3/(cik+gthemu-1)

    return step
end
