"""This module defines classes of the package."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT



abstract type RegressionModel end

# Define an object to
type LinearRegression <: RegressionModel
    function LinearRegression()
        return new()
    end
end

type LogisticRegression <: RegressionModel
    function LogisticRegression()
        return new()
    end
end



abstract type ParallelParam end

type Sequential <: ParallelParam
    function Sequential()
        return new()
    end
end

type Parallel <: ParallelParam
    function Parallel()
        return new()
    end
end





abstract type AlgoParams end

type OptParams <: AlgoParams
    itmax::Int
    stability::Float64
    learning_rate::Float64
    verbosity::Int
    function OptParams(itmax, stability, learning_rate; verbosity = 0)
        return new(itmax, stability, learning_rate, verbosity)
    end
end

type ProjParams <: AlgoParams
    ITER_MAX::Int
    precision::Float64
    sample::Int
    para_proj::ParallelParam
    para_inter::ParallelParam
    function ProjParams(ITER_MAX, precision, sample; para_proj=Sequential(), para_inter=Sequential())
        return new(ITER_MAX, precision, sample, para_proj, para_inter)
    end
end


abstract type AmbiguitySet end

type DivergenceSet <: AmbiguitySet
    function DivergenceSet()
        return new()
    end
end

type WassersteinSet <: AmbiguitySet
    function WassersteinSet()
        return new()
    end
end



abstract type GeneralConstraint end

type PositiveConstraint <: GeneralConstraint
    function PositiveConstraint()
        return new()
    end
end

type DROConstraint <: GeneralConstraint
    function DROConstraint()
        return new()
    end
end

type EntropicConstraint <: GeneralConstraint
    function EntropicConstraint()
        return new()
    end
end

abstract type DivergenceConstraint<:GeneralConstraint end

type KLConstraint <: DivergenceConstraint
    function KLConstraint()
        return new()
    end
end




type RobustModel
    descent_direction::Array{Float64,1}
    I0::UnitRange{Int64}
    name::GeneralConstraint
    regressionModel::RegressionModel
    function RobustModel(N, nb_features, ϵ, ambiguity, regressionModel)
        if ambiguity == "KLdivergence"
            descent_direction = [ϵ; 1; zeros(1:nb_features); ones(N)/N]
            I0 = 1:N
            dim = N+1
            name = KLConstraint()
        elseif ambiguity == "wasserstein"
            descent_direction = [ϵ; zeros(1:nb_features); ones(N)/N]
            I0 = 1:N^2
            dim = N^2+1
            name = DROConstraint()
        elseif ambiguity == "entropic"
            descent_direction = [1; zeros(1:nb_features); ones(N)/N]
            I0 = 1:N
            dim = N+1
            name = EntropicConstraint()
        end
        return new(descent_direction, I0, name, regressionModel)
    end
end
