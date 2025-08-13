#Ensemble Kalman Inversion: specific structures and function definitions

"""
    Inversion <: Process

An ensemble Kalman Inversion process
"""
struct Inversion <: Process end

function FailureHandler(process::Inversion, method::IgnoreFailures)
    failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens) = eki_update(ekp, u, g, y, obs_noise_cov)
    return FailureHandler{Inversion, IgnoreFailures}(failsafe_update)
end

"""
    FailureHandler(process::Inversion, method::SampleSuccGauss)

Provides a failsafe update that
 - updates the successful ensemble according to the EKI update,
 - updates the failed ensemble by sampling from the updated successful ensemble.
"""
function FailureHandler(process::Inversion, method::SampleSuccGauss)
    function failsafe_update(ekp, u, g, y, obs_noise_cov, failed_ens)
        successful_ens = filter(x -> !(x in failed_ens), collect(1:size(g, 2)))
        n_failed = length(failed_ens)
        u[:, successful_ens] =
            eki_update(ekp, u[:, successful_ens], g[:, successful_ens], y[:, successful_ens], obs_noise_cov)
        if !isempty(failed_ens)
            u[:, failed_ens] = sample_empirical_gaussian(u[:, successful_ens], n_failed)
        end
        return u
    end
    return FailureHandler{Inversion, SampleSuccGauss}(failsafe_update)
end

"""
    find_ekp_stepsize(
        ekp::EnsembleKalmanProcess{FT, IT, Inversion},
        g::AbstractMatrix{FT};
        cov_threshold::Real = 0.01,
    ) where {FT, IT}

Find largest stepsize for the EK solver that leads to a reduction of the determinant of the sample
covariance matrix no greater than cov_threshold. 
"""
function find_ekp_stepsize(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::AbstractMatrix{FT};
    cov_threshold::Real = 0.01,
) where {FT, IT}
    @assert cov_threshold > -eps(FT) "The limiting covariance reduction threshold cannot be negative."
    accept_stepsize = false

    Δt = !isempty(ekp.Δt) ? deepcopy(ekp.Δt[end]) : FT(1)

    # final_params [N_par × N_ens]
    cov_init = cov(get_u_final(ekp), dims = 2)
    while accept_stepsize == false
        ekp_copy = deepcopy(ekp)
        update_ensemble!(ekp_copy, g, Δt_new = Δt)
        cov_new = cov(get_u_final(ekp_copy), dims = 2)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt / 2
        end
    end

    return Δt

end

"""
     eki_update(
        ekp::EnsembleKalmanProcess{FT, IT, Inversion},
        u::AbstractMatrix{FT},
        g::AbstractMatrix{FT},
        y::AbstractMatrix{FT},
        obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
    ) where {FT <: Real, IT, CT <: Real}

Returns the updated parameter vectors given their current values and
the corresponding forward model evaluations, using the inversion algorithm
from eqns. (4) and (5) of Schillings and Stuart (2017).

Localization is implemented following the `ekp.localizer`.
"""
function eki_update(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    u::AbstractMatrix{FT},
    g::AbstractMatrix{FT},
    y::AbstractMatrix{FT},
    obs_noise_cov::Union{AbstractMatrix{CT}, UniformScaling{CT}},
) where {FT <: Real, IT, CT <: Real}

    # # Localization
    cov_ug = (u .- mean(u, dims = 2)) * (g .- mean(g, dims = 2))' / (size(u)[2] - 1)
    cov_gg = (g .- mean(g, dims = 2)) * (g .- mean(g, dims = 2))' / (size(g)[2] - 1)
    
    # N_obs × N_obs \ [N_obs × N_ens]
    # --> tmp is [N_obs × N_ens]
    tmp = try
        FT.((cov_gg + obs_noise_cov) \ (y - g))
    catch e
        if e isa SingularException
            LHS = Matrix{BigFloat}(cov_gg + obs_noise_cov)
            RHS = Matrix{BigFloat}(y - g)
            FT.(LHS \ RHS)
        else
            rethrow(e)
        end
    end
    return u + (cov_ug * tmp) # [N_par × N_ens]  
end

"""
    update_ensemble!(
        ekp::EnsembleKalmanProcess{FT, IT, Inversion},
        g::AbstractMatrix{FT};
        cov_threshold::Real = 0.01,
        Δt_new::Union{Nothing, FT} = nothing,
        deterministic_forward_map::Bool = true,
        failed_ens = nothing,
    ) where {FT, IT}

Updates the ensemble according to an Inversion process. 

Inputs:
 - ekp :: The EnsembleKalmanProcess to update.
 - g :: Model outputs, they need to be stored as a `N_obs × N_ens` array (i.e data are columms).
 - cov_threshold :: Threshold below which the reduction in covariance determinant results in a warning.
 - Δt_new :: Time step to be used in the current update.
 - deterministic_forward_map :: Whether output `g` comes from a deterministic model.
 - failed_ens :: Indices of failed particles. If nothing, failures are computed as columns of `g`
    with NaN entries.
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::AbstractMatrix{FT};
    cov_threshold::Real = 0.01,
    Δt_new::Union{Nothing, FT} = nothing,
    deterministic_forward_map::Bool = true,
    failed_ens = nothing,
) where {FT, IT, CT <: Real}

    #catch works when g non-square
    if !(size(g)[2] == ekp.N_ens)
        throw(
            DimensionMismatch(
                "ensemble size $(ekp.N_ens) in EnsembleKalmanProcess does not match the columns of g ($(size(g)[2])); try transposing g or check the ensemble size",
            ),
        )
    end

    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = get_u_final(ekp)

    N_obs = size(g, 1)

    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / Δt_new
    noise = rand(ekp.rng, MvNormal(zeros(N_obs), scaled_obs_noise_cov), ekp.N_ens)
    
    # Add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = deterministic_forward_map ? (ekp.obs_mean .+ noise) : (ekp.obs_mean .+ zero(noise))
    u = eki_update(ekp, u, g, y, scaled_obs_noise_cov)

    # store new parameters (and model outputs)
    pop!(ekp.u)
    push!(ekp.u, DataContainer(u, data_are_columns = true))

    return u
end
