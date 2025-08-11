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
using GaussianRandomFields
using MAT
@everywhere using Flux
@everywhere using jInv.Mesh;


# add local libraries
include("../../EnsembleKalmanProcesses/EnsembleKalmanProcesses.jl")
using .EnsembleKalmanProcesses
using .EnsembleKalmanProcesses.Observations
using .EnsembleKalmanProcesses.ParameterDistributions
using .EnsembleKalmanProcesses.DataContainers
using .EnsembleKalmanProcesses.Localizers

@everywhere include("../../FactoredEikonalFastMarching/FactoredEikonalFastMarching.jl")
@everywhere include("../../FactoredEikonalFastMarching/runAccuracyExperiment.jl");
@everywhere using .FactoredEikonalFastMarching;

# init param
order_flag = 2  # 1 or 2

# set output dir
homedir = pwd()
figure_save_directory = homedir * "/output_inclusion/"    # output results
data_save_directory = homedir * "/output_inclusion/"
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

function generate_velocity1()
    rows = 61
    cols = 61
    r = 12
    zone1_vel = 2.0
    zone2_vel = 3.0

    vel = zone1_vel * ones(rows, cols)
    centre = [31, 31]
    for i in 1:size(vel)[1]
        for j in 1:size(vel)[2]
            x = i - centre[1]
            y = j - centre[2] 
            if sqrt((x^2 + y^2)) <= r
                vel[i, j] = zone2_vel
            end
        end
    end
    vel
end

@everywhere function crosswell_double_receiver_samples_and_vcat(τ_array::Vector{Matrix{Float64}}, count::Int64)
    n = length(τ_array)
    step = 2 # div(size(τ_array[1])[1], count)
    last = size(τ_array[1])[1]  # step * count
    left_receiver_sample(τ, s, last) = [τ[i, 1] for i in 1:s:last]
    right_receiver_sample(τ, s, last) = [τ[i, end] for i in 1:s:last]
    samples = zeros(31* 2, n)
    for i in 1:n
        left = left_receiver_sample(τ_array[i], step, last)
        right = right_receiver_sample(τ_array[i], step, last)
        samples[:, i] = vcat(left[:], right[:])
    end
    reduce(vcat, samples)
end


function run()
    init_time_start = time_ns()
    # set random seed
    rng_seed = 41
    rng = Random.seed!(rng_seed)

    clean_log()

    println("******************** Velocity Model ***************************");
    vel_origin = generate_velocity1()
    vel_sampling_interval = 1
    vel = vel_origin[1:vel_sampling_interval:end, 1:vel_sampling_interval:end]
    I = [2.4, 2.4]; tmp_index = 4;
    fig_rows = 1; fig_cols = 3; single_line_target_col1 = 31; single_line_target_col2 = 31; single_line_target_col3 = 31; single_line_target_col4 = 31
    fig_size = [22.5, 6]; subplot_title_height = - 0.15; legend_height = - 0.15; 
    up_bounded_global = 3; low_bounded_global = 2; xticks_num = 5; 
    recv_marker_size = 11; recv_marker_itvl = 3; src_marker_size = 8; src_marker_itvl = 10; src_marker_offset = 0;

    vel_rows = size(vel, 1)
    vel_cols = size(vel, 2)
    vel_length = vel_rows * vel_cols    
    vel_vec = vel[:]
    h = [I[1] / vel_rows, I[2] / vel_cols]
    n = [vel_rows, vel_cols]
    colorbar_scale = 1

    # crosswell
    N_receivers = n[1]
    src = []
    for i in 1:10:n[1]
        push!(src, [i, 1])
        push!(src, [i, n[2]])
    end
    receiver_samples_and_vcat = crosswell_double_receiver_samples_and_vcat
    receivers_on_subsurface_flag = false
    receivers_in_crosswell_flag = true

    # compute groundtruth of τ
    println("\r\n|-- velocity origin forward \t-------------------------|")
    pEik_origin_T1 = [runExperimentAndWriteResults(vel, h, src[i], n, order_flag) for i in eachindex(src)]
    τ_exact = [pEik_origin_T1[i] for i in eachindex(src)]

    # initialize GRFs
    cov_tmp = CovarianceFunction(2, Matern(0.3, 2.0, σ = 1.0))
    pts1 = range(0, stop=1, length = n[1])
    pts2 = range(0, stop=1, length = n[2])
    grf = GaussianRandomField(cov_tmp, KarhunenLoeve(500), pts1, pts2)

    # Define the true values
    τ_exact_samples = receiver_samples_and_vcat(τ_exact, N_receivers)
    check_data = mean(reshape(τ_exact_samples, :, length(τ_exact)), dims=1)
    println("τ_exact_samples length: ", length(τ_exact_samples), " check_data: ", check_data)
    τ_exact_length = length(τ_exact_samples)
    vel_vec_distributions = zeros(vel_length, 1)
    grf_sample = GaussianRandomFields.sample(grf)

    # generate noise and R matrix
    u_noise = 0.002
    noise_factor = mean(check_data) * 0.05
    noise = noise_factor * randn(τ_exact_length)
    τ_exact_samples = τ_exact_samples[:] + noise[:]
    σŋ = noise_factor 
    println("σŋ: ", σŋ)
    Γ = Diagonal(σŋ^2 * ones(τ_exact_length)) 
    Γ_inv = inv(Γ)
    Γ_sqrt_inv = sqrt.(Γ_inv) 

    # Inltilize the EKI
    N_ens = 200 # number of ensemble members
    N_iter = 101 # number of EKI iterations
    vel_vec_distributions[:, 1] = grf_sample[:] # tmp[:]
    model_distribution = Samples(vel_vec_distributions)  # model_distribution = VectorOfParameterized(...)
    model_constraint = repeat([no_constraint()], vel_length) 
    model_distribution_name = "model_distribution"
    param_dict = Dict("distribution" => model_distribution, "constraint" => model_constraint, "name" => model_distribution_name)
    priors = ParameterDistribution(param_dict)
    initial_par = construct_initial_ensemble(priors, N_ens; rng_seed)
    ekiobj = EnsembleKalmanProcess(initial_par, τ_exact_samples, Γ, Inversion(), failure_handler_method = SampleSuccGauss())

    # generate the priors for GRFs
    u_stored_vecs = zeros(vel_length, N_ens) 
    for j in 1:N_ens
        grf_sample = GaussianRandomFields.sample(grf)
        u_stored_vecs[:, j] = grf_sample[:]
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

            up_bounded = up_bounded_global 
            low_bounded = low_bounded_global   
            grf_fun(grf, up, low) = sigmoid.(grf) * (up - low) .+ low

            u_n = get_u_final(ekiobj)
            u_n_array = [reshape(u_n[:, i], vel_rows, vel_cols) for i in 1:N_ens]
            
            # Evaluate forward map
            G_ens = zeros(τ_exact_length, N_ens)
            G_ens_shared = SharedArray(G_ens)
            @sync @distributed for i in 1:N_ens    
                pEik_tmp_T1 = [runExperimentAndWriteResults(grf_fun(u_n_array[i], up_bounded, low_bounded), h, src[j], n, order_flag) for j in eachindex(src)] 
                τ_tmp_f = [pEik_tmp_T1[j] for j in eachindex(src)]
                τ_tmp_f_samples = receiver_samples_and_vcat(τ_tmp_f, N_receivers)
                G_ens_shared[:, i] = τ_tmp_f_samples
            end

            G_ens = G_ens_shared.s

            handle_data_start_time = time_ns()

            # compute std
            u_n_normalized = grf_fun(u_n, up_bounded, low_bounded)
            u_n_cov_vec = [sqrt(sum((u_n_normalized[j, :] .- mean(u_n_normalized[j, :])).^2) / (N_ens - 1)) for j in 1:vel_length]
            u_n_cov_mat = reshape(u_n_cov_vec, vel_rows, vel_cols)

            u_n_mean_array_normalized = mean(u_n_normalized, dims = 2)
            u_n_mean_array_normalized = reshape(u_n_mean_array_normalized, vel_rows, vel_cols)  

            ########################################   plot result    #######################################

            # compute predicted τ
            τ_n_f = runExperimentAndWriteResults(u_n_mean_array_normalized, h, src[tmp_index], n, order_flag)  

            # compute and print current sample τ err
            τ_ARE = abs.((τ_exact[tmp_index] - τ_n_f) ./ τ_exact[tmp_index])
            τ_ARE = τ_ARE .* (τ_exact[tmp_index] .!= 0)
            τ_ARE = sum(τ_ARE)*(1 / length(τ_ARE))
            τ_metrics = sqrt(sum((τ_exact[tmp_index] - τ_n_f).^2) / sum(τ_exact[tmp_index].^2))
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
                for j in 1:recv_marker_itvl:vel_cols
                    ax[1].plot(j - 1, 0, marker = 7, markersize = recv_marker_size, 
                                markeredgecolor = "black", color = "red", clip_on = false)
                end
            end
            if receivers_in_crosswell_flag
                for j in 1:recv_marker_itvl:vel_rows
                    ax[1].plot(vel_cols - 1, j - 1, marker = 4, markersize = recv_marker_size, markeredgecolor = "black", color = "red", clip_on = false)
                end
            end
            if receivers_on_subsurface_flag
                for j in src_marker_offset:src_marker_itvl:vel_cols
                    ax[1].plot(j, src_marker_offset, marker = "o", markersize = src_marker_size,
                                markeredgecolor = "purple", color = "white", clip_on = false)
                end
            end
            if receivers_in_crosswell_flag
                for j in src_marker_offset:src_marker_itvl:vel_rows
                    ax[1].plot(src_marker_offset, j, marker = "o", markersize = src_marker_size, 
                                markeredgecolor = "purple", color = "white", clip_on = false)
                end
            end
            tmp_plt1 = ax[1].plot(0, 0, marker = "v", linestyle = "none", markersize = recv_marker_size * 0.7, 
                                  markeredgecolor = "black", color = "red", clip_on = false, label = "receivers")
            tmp_plt2 = ax[1].plot(0, 0, marker = "o", linestyle = "none", markersize = src_marker_size,
                                  markeredgecolor = "purple", color = "white", clip_on = false, label = "sources")
            ax[1].legend(bbox_to_anchor = (0, legend_height), loc = 3, fontsize = 10)
            tmp_plt1[1].set(visible = false); tmp_plt2[1].set(visible = false)
            ax[1].axis("scaled")
            ax[1].set_title("(a)", y = subplot_title_height, fontsize = 18)
            ax[1].set_xlabel("Distance (km)", fontsize = 14);
            ax[1].set_ylabel("Depth (km)", fontsize = 14);
            ax[1].set_xticks(range(0.0, vel_cols, length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 10);
            ax[1].set_yticks(range(0.0, vel_rows, length = 5), 
                            range(0.0, I[1], length = 5),
                            fontsize = 10);
            ax[1].tick_params(top = false, labeltop = false, bottom = true, labelbottom = true)
            cb1 = PyPlot.colorbar(subplot1, ax = ax[1], shrink = colorbar_scale)
            cb1.set_label("Velocity (km/s)", fontsize = 14)

            subplot2 = ax[2].matshow(u_n_mean_array_normalized, cmap="jet", norm = tmp_norm); 
            ax[2].axis("scaled")
            ax[2].set_title("(b)", y = subplot_title_height, fontsize = 18)
            ax[2].set_xlabel("Distance (km)", fontsize = 14);
            ax[2].set_ylabel("Depth (km)", fontsize = 14);
            ax[2].set_xticks(range(0.0, vel_cols, length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 10);
            ax[2].set_yticks(range(0.0, vel_rows, length = 5), 
                            range(0.0, I[1], length = 5),
                            fontsize = 10);
            ax[2].tick_params(top = false, labeltop = false, bottom = true, labelbottom = true)
            cb2 = PyPlot.colorbar(subplot2, ax = ax[2], shrink = colorbar_scale)
            cb2.set_label("Velocity (km/s)", fontsize = 14)

            subplot3 = ax[3].matshow(u_n_cov_mat, cmap="jet"); 
            ax[3].axis("scaled")
            ax[3].set_title("(c)", y = subplot_title_height, fontsize = 18)
            ax[3].set_xlabel("Distance (km)", fontsize = 14);
            ax[3].set_ylabel("Depth (km)", fontsize = 14);
            ax[3].set_xticks(range(0.0, n[2], length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 10);
            ax[3].set_yticks(range(0.0, n[1], length = 5), 
                            range(0.0, I[1], length = 5),
                            fontsize = 10);
            ax[3].tick_params(top = false, labeltop = false, bottom = true, labelbottom = true)
            cb3 = PyPlot.colorbar(subplot3, ax = ax[3], shrink = colorbar_scale)
            cb3.set_label("Std (km²/s²)", fontsize = 14)

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
            ax2.contour(τ_exact[tmp_index][end:-1:1,:],20,linestyles = "-",colors = "black",linewidths=1.0)
            ax2.contour(τ_n_f[end:-1:1,:],20,linestyles = "--",colors = "red", linewidths=1.0)
            ax2.axis("scaled")
            ax2.set_xlabel("Distance (km)", fontsize = 14);
            ax2.set_ylabel("Depth (km)", fontsize = 14);
            ax2.set_xticks(range(0.0, n[2], length = xticks_num), 
                            range(0.0, I[2], length = xticks_num),
                            fontsize = 10);
            ax2.set_yticks(range(0.0, n[1], length = 5), 
                            range(I[1], 0.0, length = 5),
                            fontsize = 10);
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
            EnsembleKalmanProcesses.update_ensemble!(ekiobj, G_ens, u_noise = u_noise, deterministic_forward_map = true, Δt_new = αₙ_inv)


            time_cost = time_ns() - time_start - handle_data_cost_time
            print_time_log("[cost time] " * string(idx), time_cost * 1e-9)
        end
    end

    println("\r\n|-- All completion! Iterations: ", idx, " \t---------|")
end

run()

