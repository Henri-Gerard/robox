"""This module defines classes of the package."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT



abstract type RegressionModel end

# Define an object to
struct LinearRegression <: RegressionModel
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function LinearRegression()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end

struct LogisticRegression <: RegressionModel
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function LogisticRegression()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end



abstract type ParallelParam end

struct Sequential <: ParallelParam
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function Sequential()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end

struct Parallel <: ParallelParam
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function Parallel()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end





abstract type AlgoParams end

struct OptParams <: AlgoParams
    itmax::Int
    stability::Float64
    learning_rate::Float64
    verbosity::Int
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function OptParams(itmax, stability, learning_rate; verbosity = 0)
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new(itmax, stability, learning_rate, verbosity)
    end
end

struct ProjParams <: AlgoParams
    ITER_MAX::Int
    precision::Float64
    sample::Int
    para_proj::ParallelParam
    para_inter::ParallelParam
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function ProjParams(ITER_MAX, precision, sample; para_proj=Sequential(), para_inter=Sequential())
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new(ITER_MAX, precision, sample, para_proj, para_inter)
    end
end


abstract type AmbiguitySet end

struct DivergenceSet <: AmbiguitySet
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function DivergenceSet()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end

struct WassersteinSet <: AmbiguitySet
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function WassersteinSet()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end



abstract type GeneralConstraint end

struct PositiveConstraint <: GeneralConstraint
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function PositiveConstraint()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end

struct DROConstraint <: GeneralConstraint
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function DROConstraint()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end

struct EntropicConstraint <: GeneralConstraint
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function EntropicConstraint()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end

abstract type DivergenceConstraint<:GeneralConstraint end

struct KLConstraint <: DivergenceConstraint
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function KLConstraint()
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        return new()
    end
end




struct RobustModel
    descent_direction::Array{Float64,1}
    I0::UnitRange{Int64}
    name::GeneralConstraint
    regressionModel::RegressionModel
    """
    Aggregator of projection

    Parameters
    ----------
    i::Int

    robustModel::RobustModel

    Returns
    -------

    """
    function RobustModel(N, nb_features, ϵ, ambiguity, regressionModel)
        """
        Aggregator of projection

        Parameters
        ----------
        i::Int

        robustModel::RobustModel

        Returns
        -------

        """
        if ambiguity == "KLdivergence"
            descent_direction = [ϵ; 1; zeros(nb_features); ones(N)/N]
            I0 = 1:N
            dim = N+1
            name = KLConstraint()
        elseif ambiguity == "wasserstein"
            descent_direction = [ϵ; zeros(nb_features); ones(N)/N]
            I0 = 1:N^2
            dim = N^2+1
            name = DROConstraint()
        elseif ambiguity == "entropic"
            descent_direction = [1; zeros(nb_features); ones(N)/N]
            I0 = 1:N
            dim = N+1
            name = EntropicConstraint()
        end
        return new(descent_direction, I0, name, regressionModel)
    end
end
