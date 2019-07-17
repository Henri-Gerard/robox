"""This module implements functions to access solution and make prediction."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT

function getsolution(x, ambiguity, nbfeatures)
    if ambiguity == "KLdivergence"
        return x[3:3+nbfeatures-1]
    elseif ambiguity ==  "wasserstein"
        return x[2:2+nbfeatures-1]
    elseif ambiguity ==  "entropic"
        return x[2:2+nbfeatures-1]
    end
end




function pred(data,θ, regression::RegressionModel)
    error("'error_loss' not defined for $(typedof(s))")
end


function pred(data, θ, regression::LinearRegression)
    pred = zeros(size(data)[1])
    for i in 1:length(pred)
        pred[i] = θ'*data[i,1:end-1]
    end
    return pred
end

function pred(data, θ, regression::LogisticRegression)
    pred = zeros(size(data)[1])
    for i in 1:length(data[:,end])
        if (1/(1+exp(θ'*data[i,1:end-1])) > 0.5)
            pred[i] = -1
        end
        if (1/(1+exp(-θ'*data[i,1:end-1])) > 0.5)
            pred[i] = 1
        end
    end
    return pred
end



function positive_rate(x, data)
    aux = zeros(size(data)[1])
    for i in 1:size(data)[1]
        aux[i] = 1/(1+exp(-x'*data[i,1:end-1]))
    end
    return aux
end
