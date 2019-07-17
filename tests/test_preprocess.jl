"""This module implements reading and initializing functions."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT

function read_data_libsvm(path, nbfeatures)
    m = readdlm(path)
    data = zeros(size(m)[1], nbfeatures+1)
    for i in 1:size(m)[1]
        for j in 2:size(m)[2]
            if m[i,j] != ""
                a,b = split(m[i,j],":")
                ai = parse(Int, a)
                bi = parse(Float64, b)
                data[i,ai] = bi
            end
        end
    end
    data[:,end] = m[:,1]
    return data
end


function create_data(name, nbfeatures, train_size, train_test_split)
    path_train = string("datasets/",name)
    data = read_data_libsvm(path_train, nbfeatures)
    xtr, xte, ytr, yte = train_test_split(data[:,1:end-1], data[:,end], train_size = train_size)
    df_train = hcat(xtr, ytr)
    df_test = hcat(xte, yte);

    return df_train, df_test
end




function initialize(Z, robustModel, constraint::GeneralConstraint)
    error("'error_loss' not defined for $(typedof(s))")
end

function initialize(Z, robustModel, constraint::KLConstraint)
    N = size(Z)[1]
    lt = size(Z)[2]-1


    x0 = rand(length(robustModel.descent_direction))
    x1 = rand(length(robustModel.descent_direction))
    x2 = ones(length(robustModel.descent_direction))
    x3 = ones(length(robustModel.descent_direction))


    x1[2] = maximum([error_loss(x1[3:3+lt-1], Z[i,:], robustModel.regressionModel) for i in 1:size(Z)[1]])
    x3[2] = maximum([error_loss(x3[3:3+lt-1], Z[i,:], robustModel.regressionModel) for i in 1:size(Z)[1]])
    for k in robustModel.I0
        aux0 = fconstraint(x0, Z, robustModel.regressionModel, k, constraint)
        aux2 = fconstraint(x2, Z, robustModel.regressionModel, k, constraint)
        if aux0 > 0
            x0[3+lt-1+k] = x0[3+lt-1+k] + aux0
        end

        if aux2 > 0
            x2[3+lt-1+k] = x2[3+lt-1+k] + aux2
        end
    end

    array_aux = [robustModel.descent_direction'*x0, robustModel.descent_direction'*x1, robustModel.descent_direction'*x2, robustModel.descent_direction'*x3]
    index = indmin(array_aux)
    if index == 1
        return x0
    elseif index == 2
        return x1
    elseif index == 3
        return x2
    elseif index == 4
        return x3
    end
end

function initialize(Z, robustModel, constraint::EntropicConstraint)
    N = size(Z)[1]
    lt = size(Z)[2]-1


    x0 = rand(length(robustModel.descent_direction))
    x1 = rand(length(robustModel.descent_direction))
    x2 = ones(length(robustModel.descent_direction))
    x3 = ones(length(robustModel.descent_direction))


    x1[1] = maximum([error_loss(x1[3:3+lt-1], Z[i,:], robustModel.regressionModel) for i in 1:size(Z)[1]])
    x3[1] = maximum([error_loss(x3[3:3+lt-1], Z[i,:], robustModel.regressionModel) for i in 1:size(Z)[1]])
    for k in robustModel.I0
        aux0 = fconstraint(x0, Z, robustModel.regressionModel, k, constraint)
        aux2 = fconstraint(x2, Z, robustModel.regressionModel, k, constraint)
        if aux0 > 0
            x0[2+lt-1+k] = x0[2+lt-1+k] + aux0
        end

        if aux2 > 0
            x2[2+lt-1+k] = x2[2+lt-1+k] + aux2
        end
    end

    array_aux = [robustModel.descent_direction'*x0, robustModel.descent_direction'*x1, robustModel.descent_direction'*x2, robustModel.descent_direction'*x3]
    index = indmin(array_aux)
    if index == 1
        return x0
    elseif index == 2
        return x1
    elseif index == 3
        return x2
    elseif index == 4
        return x3
    end
end

function initialize(Z, robustModel, constraint::DROConstraint)
    N = size(Z)[1]
    lt = size(Z)[2]-1
    x0 = rand(length(robustModel.descent_direction))
    x1 = rand(length(robustModel.descent_direction))
    x2 = rand(length(robustModel.descent_direction))
    x3 = rand(length(robustModel.descent_direction))


    for k in robustModel.I0
        aux0 = fconstraint(x0, Z, robustModel.regressionModel, k, constraint)
        aux2 = fconstraint(x2, Z, robustModel.regressionModel, k, constraint)
        if aux0 > 0
            l = div(k-1,N)+1
            x0[2+lt-1+l] = x0[2+lt-1+l] + aux0
        end
        if aux2 > 0
            l = div(k-1,N)+1
            x2[2+lt-1+l] = x2[2+lt-1+l] + aux2
        end
    end

    aux1 = maximum([error_loss(x1[2:2+lt-1], Z[i,:], robustModel.regressionModel) for i in 1:size(Z)[1]])
    for k in 2+lt:N
        x1[k] = aux1
    end

    aux3 = maximum([error_loss(x3[2:2+lt-1], Z[i,:], robustModel.regressionModel) for i in 1:size(Z)[1]])
    for k in 2+lt:N
        x3[k] = aux3
    end

    array_aux = [robustModel.descent_direction'*x0, robustModel.descent_direction'*x1, robustModel.descent_direction'*x2, robustModel.descent_direction'*x3]
    index = indmin(array_aux)
    if index == 1
        return x0
    elseif index == 2
        return x1
    elseif index == 3
        return x2
    elseif index == 4
        return x3
    end
end

function init_proj(data, robustModel, projParams)
    x0 = rand(length(robustModel.descent_direction))
    x1 = ones(length(robustModel.descent_direction))
    x0, dm, it = algo_proj(x0, data, robustModel, projParams)
    x1, dm, it = algo_proj(x1, data, robustModel, projParams)
    if robustModel.descent_direction'*x0 < robustModel.descent_direction'*x1
        return x0
    else
        return x1
    end
end
