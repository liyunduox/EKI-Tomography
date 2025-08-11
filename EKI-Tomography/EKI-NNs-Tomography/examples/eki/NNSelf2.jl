using Flux
# ENV["JULIA_CPU_THREADS"] = "auto" 
# swish elu
linear(x) = x

function create_nn_model() 
    model = [[2, 50, tanh],
             [50, 50, tanh],
             [50, 50, tanh],
             [50, 1, sigmoid],
            ]
    model
end


function create_nn_locations(rows::Int64, cols::Int64) 
    n = [rows, cols]
    locations = zeros(2, rows * cols)
    idx = 1
    for j in 1:n[2]
        for i in 1:n[1]
            locations[1, idx] = (Float64(i) - Float64(n[1]) / 2) / Float64(n[1]) * 2.0
            locations[2, idx] = (Float64(j) - Float64(n[2]) / 2) / Float64(n[2]) * 2.0
            idx = idx + 1
        end
    end
    locations
end

function generate_nn_random_params(model::Vector{Vector{Any}}) 
    layers_num = length(model)

    pars = Array{Array{Any}}(undef, 2)
    Ws = pars[1] = Array{Any}(undef, layers_num)
    bs = pars[2] = Array{Any}(undef, layers_num)

    for l in 1:layers_num
        rows = model[l][2]
        cols = model[l][1]
        Ws[l] = 0.1 * randn(rows, cols)    # circular: 0.1  arched：0.1   
        bs[l] = 0.1 * randn(rows, 1)
    end
    pars
end

function generate_nn_random_fields(locations, pars::Array{Array{Any}}, model::Vector{Vector{Any}})
    layers_num = length(model)

    
    Ws = pars[1]
    bs = pars[2]
    x = locations
    for l in 1:layers_num
        x = model[l][3].(Ws[l] * x .+ bs[l])
    end

    x
end

function nn_params_to_vec(pars::Array{Array{Any}}) 
    Ws = pars[1]
    bs = pars[2]
    
    layers_num = length(Ws)
    tmp = []
    for l in 1:layers_num
        push!(tmp, Ws[l][:])
        push!(tmp, bs[l][:])
    end
    pars_vec = vcat(tmp...)
    pars_vec
end

function get_nn_params_length(model::Vector{Vector{Any}}) 
    layers_num = length(model)

    params_length = 0
    for l in 1:layers_num
        rows = model[l][2]
        cols = model[l][1]
        params_length += rows * cols + rows
    end
    params_length
end

function nn_vec_to_params(vec::Array{Float64}, model::Vector{Vector{Any}}) 
    layers_num = length(model)

    pars = Array{Array{Any}}(undef, 2)
    Ws = pars[1] = Array{Any}(undef, layers_num)
    bs = pars[2] = Array{Any}(undef, layers_num)
    
    data_start = 1
    for l in 1:layers_num
        rows = model[l][2] 
        cols = model[l][1]
        data_end = rows * cols + data_start - 1
        Ws[l] = reshape(vec[data_start:data_end], rows, cols)

        data_start = data_end + 1
        data_end = rows + data_start - 1
        if rows == 1
            bs[l] = vec[data_end] * ones(1, 1)
        else
            bs[l] = reshape(vec[data_start:data_end], rows, 1)
        end
        data_start = data_end + 1
    end
    pars
end

function test()
    n = [52, 252]
    locations = create_nn_locations(n[1], n[2])
    model1 = create_nn_model()

    count = 1
    pars_length = get_nn_params_length(model1)
    n_pars_array = zeros(pars_length, count)
    for j in 1:count
        pars = nn_params_to_vec(generate_nn_random_params(model1))
        n_pars_array[:, j] = pars
    end
    pars_mean = mean(n_pars_array, dims = 2)
    nnRFs = generate_nn_random_fields(locations, nn_vec_to_params(pars_mean, model1), model1)
    matshow(reshape(nnRFs, n[1], n[2]), cmap = "jet"); colorbar()
end



