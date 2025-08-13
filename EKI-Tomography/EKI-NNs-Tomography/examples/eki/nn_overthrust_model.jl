# add official libraries 
using Distributed
addprocs(10)
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using JLD2
using Random
using Printf
using Statistics
using SharedArrays
using PyPlot
using MAT
using Flux
@everywhere using jInv.Mesh;


# add local libraries
include("../../EnsembleKalmanProcesses/EnsembleKalmanProcesses.jl")
using .EnsembleKalmanProcesses
using .EnsembleKalmanProcesses.Observations
using .EnsembleKalmanProcesses.ParameterDistributions
using .EnsembleKalmanProcesses.DataContainers
using .EnsembleKalmanProcesses.Localizers

@everywhere include("./NNSelf2.jl")
@everywhere include("../../FactoredEikonalFastMarching/FactoredEikonalFastMarching.jl")
@everywhere include("../../FactoredEikonalFastMarching/runAccuracyExperiment.jl");
@everywhere using .FactoredEikonalFastMarching;

# init param
order_flag = 2  # 1 or 2

# set output dir
homedir = pwd()
figure_save_directory = homedir * "/output_overthrust/"    # output results
data_save_directory = homedir * "/output_overthrust/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end

function clean_log()
    file_path = figure_save_directory * "/1_log.txt"
    io = open(file_path, "w")
    write(io, "")
    close(io)

    file_path = figure_save_directory * "/1_time.txt"
    io = open(file_path, "w")
    write(io, "")
    close(io)
end

function print_log(str::String)
    println(str)

    file_path = figure_save_directory * "/1_log.txt"
    io = open(file_path, "a")
    str = str * "\r\n"
    write(io, str)
    close(io)
end

function print_time_log(head::String, time::Float64)
    file_path = figure_save_directory * "/1_time.txt"
    io = open(file_path, "a")
    str = head * ": " * string(time) * "\r\n"
    print(str)
    write(io, str)
    close(io)
end

function generate_velocity2()
    v0 = 2
    vergrad = 2.0
    horgrad = 0.0

    zmin = 0.0; zmax = 1.0; deltaz = 0.02;
    xmin = 0.0; xmax = 4.8; deltax = 0.02;

    sz = 1.0; sx = 1.0

    z = range(zmin, zmax, step = deltaz)
    nz = z.len

    x = range(xmin, xmax, step = deltax)
    nx = x.len

    Z = first.(Iterators.product(z,x))
    X = last.(Iterators.product(z,x))

    vs = v0 + vergrad * sz + horgrad * sx 
    res1 = ((X .< 2.4) .&& (X .> 1.4)) 
    res2 = ((X .< 3.4) .&& (X .>= 2.4)) 
    velmodel = vs .+ vergrad*(Z .- sz) + horgrad*(X .- sx) + 2 * res1 .* (2.25 .- 1/3*x.^3 + 2*x.^2 - 3.75*x)' + 2 * res2 .* (2.25 .- 1/3*(4.8 .-x).^3 + 2*(4.8 .-x).^2 - 3.75*(4.8 .-x))'
    velmodel
end

@everywhere function crosswell_receiver_samples_and_vcat2(τ_array::Vector{Matrix{Float64}}, count::Int64)
    n = length(τ_array)
    step1 = 1
    last1 = size(τ_array[1])[2] 
    top_receiver_sample(τ, s, last) = [τ[1, i] for i in 1:s:last]
    samples = zeros(last1, n)
    for i in 1:n
        top = top_receiver_sample(τ_array[i], step1, last1)
        samples[:, i] = top 
    end
    reduce(vcat, samples)
end

function read_model_from_mat_transpose(path::String, vername::String; velscale::Float64 = 1.0)
    file = matopen(path)
    model = read(file, vername) * velscale
    return copy(model')
end

function run()
    init_time_start = time_ns()
    # set random seed
    rng_seed = 41
    rng = Random.seed!(rng_seed)

    clean_log()

    println("******************** Velocity Model ***************************");
    vel_origin = read_model_from_mat_transpose(homedir * "/model/overthrust.mat", "y400"; velscale = 0.001)
    vel_origin = vel_origin[1:181, :]
    vel_origin = vel_origin[1:2:end, 1:2:end]
    vel_sampling_interval = 2
    vel = vel_origin[1:vel_sampling_interval:end, 1:vel_sampling_interval:end]
    I = [3.6, 16.0]; src_ref = [1, 201];
    fig_rows = 3; fig_cols = 1; single_line_target_col1 = 60; single_line_target_col2 = 120; single_line_target_col3 = 180; single_line_target_col4 = 288;
    fig_size = [15, 11]; subplot_title_height = - 0.32; legend_height = - 0.32;   
    up_bounded_global = 6.0; low_bounded_global = 2.35; xticks_num = 9; 
    recv_marker_size = 11; recv_marker_itvl = 10; src_marker_size = 8; src_marker_itvl = 45; src_marker_offset = 2;

    vel_origin_rows = size(vel_origin, 1)
    vel_origin_cols = size(vel_origin, 2)
    vel_origin_length = vel_origin_rows * vel_origin_cols
    vel_rows = size(vel, 1)
    vel_cols = size(vel, 2)
    vel_length = vel_rows * vel_cols    # length(vel)
    vel_vec = vel[:]
    h = [I[1] / (vel_rows-1), I[2] / (vel_cols-1)]
    h_origin = [I[1] / (vel_origin_rows-1), I[2] / (vel_origin_cols-1)]
    n = [vel_rows, vel_cols]
    n_origin = [vel_origin_rows, vel_origin_cols]
    colorbar_scale = 1

    # subsurface
    N_receivers = n[2]
    src = [[1, i] for i in 1:10:n[2]]    
    src_origin = [[1, i] for i in 1:20:n_origin[2]]    
    receiver_samples_and_vcat = crosswell_receiver_samples_and_vcat2
    receivers_on_subsurface_flag = true
    receivers_in_crosswell_flag = false

    # compute groundtruth of τ
    println("\r\n|-- velocity origin forward \t-------------------------|")
    pEik_origin_T1 = [runExperimentAndWriteResults(vel_origin, h_origin, src_origin[i], n_origin, order_flag) for i in eachindex(src_origin)]
    τ_exact = [pEik_origin_T1[i][1:vel_sampling_interval:end, 1:vel_sampling_interval:end] for i in eachindex(src)]
    τ_exact_ref = runExperimentAndWriteResults(vel_origin, h_origin, src_ref, n_origin, order_flag)
   
    # Define the true values
    τ_exact_samples = receiver_samples_and_vcat(τ_exact, N_receivers)
    check_data = mean(reshape(τ_exact_samples, :, length(τ_exact)), dims=1)
    println("τ_exact_samples length: ", length(τ_exact_samples), " check_data: ", check_data)
    τ_exact_length = length(τ_exact_samples)

    # initialize NNs
    nn_model1 = create_nn_model()
    normalized_locations_origin = create_nn_locations(size(vel_origin)[1], size(vel_origin)[2])
    normalized_locations = create_nn_locations(vel_rows, vel_cols)

    # Ensemble Kalman Inversion
    nn_pars = generate_nn_random_params(nn_model1)
    nn_pars_vec = nn_params_to_vec(nn_pars)
    nn_pars_vec_length = length(nn_pars_vec)
    vel_vec_distributions = zeros(nn_pars_vec_length, 1)
    vel_vec_distributions[:, 1] = nn_pars_vec[:]
    
    # generate noise and R matrix
    N_ens = 200 # number of ensemble members
    N_iter = 101 # number of EKI iterations
    u_noise = 0.02
    noise_factor = mean(check_data) * 0.05
    noise = noise_factor * randn(τ_exact_length)
    τ_exact_samples = τ_exact_samples[:] + noise[:]
    σŋ = noise_factor 
    println("σŋ: ", σŋ)
    Γ = Diagonal(σŋ^2 * ones(τ_exact_length)) 
    Γ_inv = inv(Γ)
    Γ_sqrt_inv = sqrt.(Γ_inv) 
     
    # Inltilize the EKI
    model_distribution = Samples(vel_vec_distributions)  # model_distribution = VectorOfParameterized(...)
    model_constraint = repeat([no_constraint()], nn_pars_vec_length) 
    model_distribution_name = "model_distribution"
    param_dict = Dict("distribution" => model_distribution, "constraint" => model_constraint, "name" => model_distribution_name)
    priors = ParameterDistribution(param_dict)
    initial_par = construct_initial_ensemble(priors, N_ens; rng_seed)
    ekiobj = EnsembleKalmanProcess(initial_par, τ_exact_samples, Γ, Inversion(), failure_handler_method = SampleSuccGauss())

    # generate the priors for NNs
    u_stored_vecs = zeros(nn_pars_vec_length, N_ens) 
    for j in 1:N_ens
        nn_pars = generate_nn_random_params(nn_model1)
        nn_pars_vec = nn_params_to_vec(nn_pars)
        u_stored_vecs[:, j] = nn_pars_vec[:]
    end
    pop!(ekiobj.u)
    push!(ekiobj.u, DataContainer(u_stored_vecs, data_are_columns = true))

    αₙ = 1
    
    vel_err_array = []
    s_τ_err_array = []

    w_interval = 20
    w_index = 1
    diff_ratio_threshold = 0.05
    diff = zeros(w_interval)

    mean_window_size = 5
    mean_window_index = 1
    mean_window = zeros(mean_window_size)

    init_time_cost = time_ns() - init_time_start
    print_time_log("[cost time] init", init_time_cost * 1e-9)

    idx = 0
    # EKI iterations  
    for iter in 1:N_iter
        begin #@time begin
            time_start = time_ns()
            idx = iter - 1
            PyPlot.close("all")
            print_log("\r\n|-- EKI iteration: " * string(idx) * " \t-------------------------|")

            u_n = get_u_final(ekiobj)
            Q_covariance = Diagonal(u_noise^2 * ones(nn_pars_vec_length)) 
            u_p_noise = rand(rng, MvNormal(zeros(nn_pars_vec_length), Q_covariance), N_ens)
            u_n = u_n + u_p_noise
            pop!(ekiobj.u)
            push!(ekiobj.u, DataContainer(u_n, data_are_columns = true))

            u_n_array_vec = zeros(vel_rows * vel_cols,  N_ens)
            u_n_array_vec_shared = SharedArray(u_n_array_vec)
            @sync @distributed for j in 1:N_ens
                up_bounded = up_bounded_global 
                low_bounded = low_bounded_global
                nnRFs = generate_nn_random_fields(normalized_locations, nn_vec_to_params(u_n[:, j], nn_model1), nn_model1)
                nnRFs = nnRFs * (up_bounded - low_bounded) .+ low_bounded
                u_n_array_vec_shared[:, j] = nnRFs
            end
            u_n_array_vec = u_n_array_vec_shared.s
            u_n_array = [reshape(u_n_array_vec[:, j], vel_rows, vel_cols) for j in 1:N_ens]
 
            # Evaluate forward map
            G_ens = zeros(τ_exact_length, N_ens)
            G_ens_shared = SharedArray(G_ens)
            @sync @distributed for i in 1:N_ens       
                pEik_tmp_T1 = [runExperimentAndWriteResults(u_n_array[i], h, src[j], n, order_flag) for j in eachindex(src)] 
                τ_tmp_f = [pEik_tmp_T1[j] for j in eachindex(src)]
                τ_tmp_f_samples = receiver_samples_and_vcat(τ_tmp_f, N_receivers)
                G_ens_shared[:, i] = τ_tmp_f_samples
            end

            G_ens = G_ens_shared.s

            handle_data_start_time = time_ns()

            # compute std
            u_n_array_vec_origin = zeros(vel_origin_rows * vel_origin_cols,  N_ens)
            u_n_array_vec_shared_origin = SharedArray(u_n_array_vec_origin)
            @sync @distributed for j in 1:N_ens
                up_bounded = up_bounded_global 
                low_bounded = low_bounded_global
                nnRFs = generate_nn_random_fields(normalized_locations_origin, nn_vec_to_params(u_n[:, j], nn_model1), nn_model1)
                nnRFs = nnRFs * (up_bounded - low_bounded) .+ low_bounded
                u_n_array_vec_shared_origin[:, j] = nnRFs
            end
            u_n_array_vec_origin = u_n_array_vec_shared_origin.s
            u_n_mean_array_normalized = mean(u_n_array_vec_origin, dims = 2)
            u_n_mean_array_normalized = reshape(u_n_mean_array_normalized, vel_origin_rows, vel_origin_cols)  

            # u_n_cov_vec = diag(cov(u_n, u_n, dims=2))
            u_n_cov_vec = [sqrt(sum((u_n_array_vec_origin[j, :] .- mean(u_n_array_vec_origin[j, :])).^2) / (N_ens - 1)) for j in 1:vel_origin_length]
            u_n_cov_mat = reshape(u_n_cov_vec, vel_origin_rows, vel_origin_cols)

            ########################################   plot result    #######################################

            # plot 95% interval
            Plots.theme(:ggplot2)
            plot_x = 0:0.04:I[1]
            true_single_line = vel_origin[:, single_line_target_col1]
			mean_single_line = u_n_mean_array_normalized[:, single_line_target_col1]
            std_single_line = u_n_cov_mat[:, single_line_target_col1]

            Plots.plot(plot_x, true_single_line, label = "Ground-truth")
            Plots.plot!(plot_x, mean_single_line, ribbon = 1.96 * std_single_line, label = "Posterior mean", legend=:topleft)
            Plots.plot!(xlabel = "Z (km)", ylabel = "Velocity (km/s)", legend=:topleft)
            Plots.plot!(size=(600, 300))
            plot_name = data_save_directory * "singleStd$idx 0 .png"
            Plots.savefig(plot_name)      # saves the CURRENT_PLOT as a .png

            true_single_line = vel_origin[:, single_line_target_col2]
			mean_single_line = u_n_mean_array_normalized[:, single_line_target_col2]
            std_single_line = u_n_cov_mat[:, single_line_target_col2]

            Plots.plot(plot_x, true_single_line, label = "Ground-truth")
            Plots.plot!(plot_x, mean_single_line, ribbon = 1.96 * std_single_line, label = "Posterior mean", legend=:topleft)
            Plots.plot!(xlabel = "Z (km)", ylabel = "Velocity (km/s)", legend=:topleft)
            Plots.plot!(size=(600, 300))
            plot_name = data_save_directory * "singleStd$idx 1.png"
            Plots.savefig(plot_name)      # saves the CURRENT_PLOT as a .png

            true_single_line = vel_origin[:, single_line_target_col3]
			mean_single_line = u_n_mean_array_normalized[:, single_line_target_col3]
            std_single_line = u_n_cov_mat[:, single_line_target_col3]

            Plots.plot(plot_x, true_single_line, label = "Ground-truth")
            Plots.plot!(plot_x, mean_single_line, ribbon = 1.96 * std_single_line, label = "Posterior mean", legend=:topleft)
            Plots.plot!(xlabel = "Z (km)", ylabel = "Velocity (km/s)", legend=:topleft)
            Plots.plot!(size=(600, 300))
            plot_name = data_save_directory * "singleStd$idx 2.png"
            Plots.savefig(plot_name)      # saves the CURRENT_PLOT as a .png

			true_single_line = vel_origin[:, single_line_target_col4]
			mean_single_line = u_n_mean_array_normalized[:, single_line_target_col4]
            std_single_line = u_n_cov_mat[:, single_line_target_col4]

            Plots.plot(plot_x, true_single_line, label = "Ground-truth")
            Plots.plot!(plot_x, mean_single_line, ribbon = 1.96 * std_single_line, label = "Posterior mean")
            Plots.plot!(xlabel = "Z (km)", ylabel = "Velocity (km/s)", legend=:topleft)
            Plots.plot!(size=(600, 300))
            plot_name = data_save_directory * "singleStd$idx 3.png"
            Plots.savefig(plot_name)    

            # compute predicted τ
            τ_n_f = runExperimentAndWriteResults(u_n_mean_array_normalized, h_origin, src_ref, n_origin, order_flag)  

            # compute and print current sample τ err
            τ_ARE = abs.((τ_exact_ref - τ_n_f) ./ τ_exact_ref)
            τ_ARE = τ_ARE .* (τ_exact_ref .!= 0)
            τ_ARE = sum(τ_ARE)*(1 / length(τ_ARE))
            τ_metrics = sqrt(sum((τ_exact_ref - τ_n_f).^2) / sum(τ_exact_ref.^2))
            τ_err = Γ_sqrt_inv * (τ_exact_samples - mean(G_ens, dims = 2))
            sqrtN = sqrt(length(τ_err))  
            τ_err_norm_2 = norm(τ_err, 2) / sqrtN
            τ_err_norm_2 = τ_err_norm_2^2
            push!(s_τ_err_array, τ_err_norm_2) 
            print_log("Sampled traveltime loss: " * string(τ_err_norm_2))
            print_log("traveltime ARE: " * string(τ_ARE))
            print_log("traveltime metrics: " * string(τ_metrics))
            
            # compute and print current model err
            vel_ARE = abs.((vel_origin[:] - u_n_mean_array_normalized[:]) ./ vel_origin[:])
			vel_ARE = sum(vel_ARE)*(1 / length(vel_origin[:]))
            vel_metrics = sqrt(sum((vel_origin[:] - u_n_mean_array_normalized[:]).^2) / sum(vel_origin[:].^2))
            tmp_err = (vel_origin[:] - u_n_mean_array_normalized[:])
            sqrtN = sqrt(length(tmp_err))
            vel_err_norm_2 = norm(tmp_err, 2) / sqrtN
            vel_err_norm_2 = vel_err_norm_2^2
            push!(vel_err_array, vel_err_norm_2) 
            print_log("Velocity loss: " * string(vel_err_norm_2))
            print_log("Velocity ARE: " * string(vel_ARE))
            print_log("Velocity metrics: " * string(vel_metrics))

            tmp_err_array = [s_τ_err_array  vel_err_array]'

            @save data_save_directory * "dataIter$idx.jld2" u_n
            @save data_save_directory * "unCov$idx.jld2" u_n_cov_mat
            @save data_save_directory * "timeIter$idx.jld2" τ_n_f
            @save data_save_directory * "0_error.jld2" tmp_err_array

            # plot loss
            err_fig, err_ax = plt.subplots(2, 1)
            err_ax[1].semilogy(0:idx, tmp_err_array[1, :], "r")
            err_ax[2].semilogy(0:idx, tmp_err_array[2, :], "b")
            err_ax[1].legend(["Sampled traveltimes error"])
            err_ax[2].legend(["Velocity error"])
            tmp_xticks_interval = 10
            err_ax[1].set_xticks(0:tmp_xticks_interval:100);
            err_ax[2].set_xticks(0:tmp_xticks_interval:100);
            err_ax[1].set_xlabel("Iterations", fontsize = 14);
            err_ax[1].set_ylabel("Loss", fontsize = 14);
            err_ax[2].set_xlabel("Iterations", fontsize = 14);
            err_ax[2].set_ylabel("Loss", fontsize = 14);
            err_fig.align_ylabels()
            err_fig.tight_layout()
            plot_name = data_save_directory * "0_err.png"
            PyPlot.savefig(plot_name, dpi = 600, bbox_inches = "tight", pad_inches = 0.1)

            fig, ax = plt.subplots(fig_rows, fig_cols, figsize = fig_size)

            tmp_norm = matplotlib.colors.Normalize(vmin = minimum([vel_origin u_n_mean_array_normalized]), 
                                                        vmax = maximum([vel_origin u_n_mean_array_normalized])) 

            # plot all result (groundtruth, inversion result, std)                                       
            subplot1 = ax[1].matshow(vel_origin, cmap="jet", norm = tmp_norm);
            if receivers_on_subsurface_flag
                tmp_x = []
                tmp_y = []
                for j in 1:recv_marker_itvl:vel_origin_cols
                    push!(tmp_x, j - 1)
                    push!(tmp_y, 0)
                end
                # for j in 1:recv_marker_itvl:vel_origin_cols
                #     ax[1].plot(j - 1, 0, marker = 7, markersize = recv_marker_size, 
                #                 markeredgecolor = "black", color = "red", clip_on = false)
                # end
                ax[1].plot(tmp_x, tmp_y, marker = 7, markersize = recv_marker_size, linestyle = "none",
                                markeredgecolor = "black", color = "red", clip_on = false)
            end
            if receivers_in_crosswell_flag
                # for j in 1:recv_marker_itvl:vel_origin_rows
                #     ax[1].plot(vel_origin_cols - 1, j - 1, marker = 4, markersize = recv_marker_size, markeredgecolor = "black", color = "red", clip_on = false)
                # end
                tmp_x = []
                tmp_y = []
                for j in 1:recv_marker_itvl:vel_origin_rows
                    push!(tmp_x, vel_origin_cols - 1)
                    push!(tmp_y, j - 1)
                end
                ax[1].plot(tmp_x, tmp_y, marker = 4, markersize = recv_marker_size, linestyle = "none", 
                           markeredgecolor = "black", color = "red", clip_on = false)
            end
            if receivers_on_subsurface_flag
                # for j in src_marker_offset:src_marker_itvl:vel_origin_cols
                #     ax[1].plot(j, src_marker_offset, marker = "o", markersize = src_marker_size,
                #                 markeredgecolor = "purple", color = "white", clip_on = false)
                # end
                tmp_x = []
                tmp_y = []
                for j in src_marker_offset:src_marker_itvl:vel_origin_cols
                    push!(tmp_x, j)
                    push!(tmp_y, src_marker_offset)
                end
                ax[1].plot(tmp_x, tmp_y, marker = "o", markersize = src_marker_size, linestyle = "none", 
                                markeredgecolor = "purple", color = "white", clip_on = false)
            end
            if receivers_in_crosswell_flag
                # for j in src_marker_offset:src_marker_itvl:vel_origin_rows
                #     ax[1].plot(src_marker_offset, j, marker = "o", markersize = src_marker_size, 
                #                 markeredgecolor = "purple", color = "white", clip_on = false)
                # end
                tmp_x = []
                tmp_y = []
                for j in src_marker_offset:src_marker_itvl:vel_origin_rows
                    push!(tmp_x, src_marker_offset)
                    push!(tmp_y, j)
                end
                ax[1].plot(tmp_x, tmp_y, marker = "o", markersize = src_marker_size, linestyle = "none",  
                           markeredgecolor = "purple", color = "white", clip_on = false)  
            end
            tmp_plt1 = ax[1].plot(0, 0, marker = "v", linestyle = "none", markersize = recv_marker_size * 0.7, 
                                  markeredgecolor = "black", color = "red", clip_on = false, label = "receivers")
            tmp_plt2 = ax[1].plot(0, 0, marker = "o", linestyle = "none", markersize = src_marker_size,
                                  markeredgecolor = "purple", color = "white", clip_on = false, label = "sources")
            ax[1].legend(bbox_to_anchor = (0, legend_height), loc = 3, fontsize = 10)
            tmp_plt1[1].set(visible = false); tmp_plt2[1].set(visible = false)
            ax[1].axis("scaled")
            ax[1].set_title("(a)", y = subplot_title_height, fontsize = 20)
            ax[1].set_xlabel("Distance (km)", fontsize = 18);
            ax[1].set_ylabel("Depth (km)", fontsize = 18);
            ax[1].set_xticks(range(0.0, vel_origin_cols, length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 14);
            ax[1].set_yticks(range(0.0, vel_origin_rows, length = 5), 
                            range(0.0, I[1], length = 5),
                            fontsize = 14);
            ax[1].tick_params(top = false, labeltop = false, bottom = true, labelbottom = true)
            cb1 = PyPlot.colorbar(subplot1, ax = ax[1], shrink = colorbar_scale)
            cb1.set_label("Velocity (km/s)", fontsize = 18)

            subplot2 = ax[2].matshow(u_n_mean_array_normalized, cmap="jet", norm = tmp_norm); 
            ax[2].axis("scaled")
            ax[2].set_title("(b)", y = subplot_title_height, fontsize = 20)
            ax[2].set_xlabel("Distance (km)", fontsize = 18);
            ax[2].set_ylabel("Depth (km)", fontsize = 18);
            ax[2].set_xticks(range(0.0, vel_origin_cols, length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 14);
            ax[2].set_yticks(range(0.0, vel_origin_rows, length = 5), 
                            range(0.0, I[1], length = 5),
                            fontsize = 14);
            ax[2].tick_params(top = false, labeltop = false, bottom = true, labelbottom = true)
            cb2 = PyPlot.colorbar(subplot2, ax = ax[2], shrink = colorbar_scale)
            cb2.set_label("Velocity (km/s)", fontsize = 18)

            subplot3 = ax[3].matshow(u_n_cov_mat, cmap="jet"); 
            ax[3].axis("scaled")
            ax[3].set_title("(c)", y = subplot_title_height, fontsize = 20)
            ax[3].set_xlabel("Distance (km)", fontsize = 18);
            ax[3].set_ylabel("Depth (km)", fontsize = 18);
            ax[3].set_xticks(range(0.0, n[2], length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 14);
            ax[3].set_yticks(range(0.0, n[1], length = 5), 
                            range(0.0, I[1], length = 5),
                            fontsize = 14);
            ax[3].tick_params(top = false, labeltop = false, bottom = true, labelbottom = true)
            cb3 = PyPlot.colorbar(subplot3, ax = ax[3], shrink = colorbar_scale)
            cb3.set_label("STD (km/s)", fontsize = 18)

            fig.tight_layout()
            plot_name = data_save_directory * "uAll$idx.png"
            PyPlot.savefig(plot_name, dpi = 600, bbox_inches = "tight", pad_inches = 0.1)

            # plot velocity error
            PyPlot.matshow(abs.(vel_origin - u_n_mean_array_normalized), cmap = "jet"); colorbar(label = "Velocity (km/s)");
            PyPlot.axis("scaled")
            PyPlot.xlabel("Distance (km)", fontsize = 18);
            PyPlot.ylabel("Depth (km)", fontsize = 18);
            PyPlot.xticks(range(0.0, n[2], length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 14);
            PyPlot.yticks(range(0.0, n[1], length = 5), 
                            range(0.0, I[1], length = 5),
                            fontsize = 14);
            PyPlot.tick_params(top = false, labeltop = false, bottom = true, labelbottom = true)
            plot_name = data_save_directory * "error$idx.png"
            PyPlot.savefig(plot_name, dpi = 600, bbox_inches = "tight", pad_inches = 0.1)

            # plot current τ_ture and τ_forward
            fig2, ax2 = plt.subplots(1, 1)
            ax2.contour(τ_exact_ref[end:-1:1,:],20,linestyles = "-",colors = "black",linewidths=1.0)
            ax2.contour(τ_n_f[end:-1:1,:],20,linestyles = "--",colors = "red", linewidths=1.0)
            ax2.axis("scaled")
            # ax2.set_title("(a)", y = subplot_title_height, fontsize = 18)
            ax2.set_xlabel("Distance (km)", fontsize = 18);
            ax2.set_ylabel("Depth (km)", fontsize = 18);
            ax2.set_xticks(range(0.0, n_origin[2], length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 14);
            ax2.set_yticks(range(0.0, n_origin[1], length = 5), 
                            range(I[1], 0.0, length = 5),
                            fontsize = 14);
            fig2.tight_layout()
            plot_name = data_save_directory * "timesAll$idx.png"
            PyPlot.savefig(plot_name, dpi = 600, bbox_inches = "tight", pad_inches = 0.1)

            ############################ plot end ##############################################

            handle_data_cost_time = time_ns() - handle_data_start_time
                   
           
            # stop criterion
            if mean_window_index > mean_window_size 
                mean_window_index = 1  
            end
                    
            mean_window[mean_window_index] = norm(Γ_sqrt_inv * (τ_exact_samples - mean(G_ens, dims = 2)), 2) / sqrt(length(τ_exact_samples))
            mean_window_index = mean_window_index + 1 
            mean_window_occupied_size = idx + 1
            if mean_window_occupied_size > mean_window_size 
                mean_window_occupied_size = mean_window_size
            end
            diff_i = sum(mean_window) / mean_window_occupied_size
        
            if idx >= w_interval 
                diff_ratio_all = abs.(diff .- diff_i) / diff_i
                diff_ratio_max = maximum(diff_ratio_all)
                print_log("diff_ratio_max: " * string(diff_ratio_max))
                if diff_ratio_max < diff_ratio_threshold                     
                    break 
                end
            end
                                  
            if w_index > w_interval 
                w_index = 1
            end
                
            diff[w_index] = diff_i   
            w_index += 1

            # updates EKI
            αₙ = 1
            αₙ_inv = αₙ \ 1.0 
            EnsembleKalmanProcesses.update_ensemble!(ekiobj, G_ens, deterministic_forward_map = true, Δt_new = αₙ_inv)


            time_cost = time_ns() - time_start - handle_data_cost_time
            print_time_log("[cost time] " * string(idx), time_cost * 1e-9)
        end
    end

    println("\r\n|-- All completion! Iterations: ", idx, " \t---------|")
end

run()

