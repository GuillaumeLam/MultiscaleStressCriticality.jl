module MultiscaleFramework

using Random, LinearAlgebra, DifferentialEquations, StatsBase, Statistics, FFTW, Printf
# using PyPlot, 
using PyPlot, GLMakie, FileIO, Colors, GeometryBasics

###############################################################################
# 1) Sub-Level ODE Functions
###############################################################################

"""
    hkb_ode_level!(du, u, p_scale, offset, t)

Updates the HKB (Haken-Kelso-Bunz) oscillator states for the "behavior" level.
Suppose we have n_b phases: φ_1,..., φ_n_b, stored in u[offset].

We also allow for within-level adjacency or cross-level influences if needed.
p_scale is a NamedTuple with relevant parameters for HKB:
   :ω   => vector of intrinsic frequencies
   :a,b => coupling strengths for sin(φj-φi) and sin(2(φj-φi))
   :A   => adjacency matrix for behavior-behavior coupling
"""
function hkb_ode_level!(du, u, p_scale, offset, t)
    ω    = p_scale.ω
    a    = p_scale.a
    b    = p_scale.b
    A    = p_scale.A  # adjacency, size n_b x n_b

    n_b = length(ω)
    # The state for behavior is φ_i in u[offset : offset+n_b-1]
    # We'll do pairwise sums for HKB coupling
    for i in 1:n_b
        idx_i = offset + (i-1)
        φi = u[idx_i]
        # basic: dφi/dt = ωi
        dφ = ω[i]

        # add HKB sums
        for j in 1:n_b
            # if j != i && A[i,j] != 0.0
            if j != i && j ≤ size(A, 2) && i ≤ size(A, 1) && A[i, j] != 0.0
                idx_j = offset + (j-1)
                φj = u[idx_j]
                Δ = φj - φi
                dφ += A[i,j]*( a*sin(Δ) + b*sin(2Δ) )
            end
        end

        # store in du
        du[idx_i] = dφ
    end
end

"""
    sl_ode_level!(du, u, p_scale, offset, t)

Updates the Stuart–Landau oscillator states for the "neural" level.
We have n_n oscillators, each with real/imag parts: (x_k, y_k).

p_scale NamedTuple might have:
   :μ   => real growth param
   :ω   => array of frequencies
   :K   => coupling strength
   :A   => adjacency n_n x n_n
"""
function sl_ode_level!(du, u, p_scale, offset, t)
    μ    = p_scale.μ
    ω    = p_scale.ω
    K    = p_scale.K
    A    = p_scale.A

    n_n = length(ω)
    # state for neural is stored in 2n_n variables:
    #  (x1, y1, x2, y2, ..., x_n_n, y_n_n)
    for k in 1:n_n
        # index in big vector
        xk_idx = offset + 2(k-1) + 1
        yk_idx = offset + 2(k-1) + 2
        xk = u[xk_idx]
        yk = u[yk_idx]
        zk = xk + im*yk
        r2 = abs2(zk)

        # basic SL eq: dz/dt = (μ + iω_k)z - |z|^2 z + coupling
        # we separate real & imag
        growth_re = μ*xk - ω[k]*yk
        growth_im = ω[k]*xk + μ*yk
        nonlin_re = -r2*xk
        nonlin_im = -r2*yk

        # coupling from adjacency
        Cx, Cy = 0.0, 0.0
        for j in 1:n_n
            if j != k && A[k,j] != 0.0
                xj_idx = offset + 2(j-1) + 1
                yj_idx = offset + 2(j-1) + 2
                dx = u[xj_idx] - xk
                dy = u[yj_idx] - yk
                Cx += A[k,j]*dx
                Cy += A[k,j]*dy
            end
        end
        # scale by K
        coup_re = K * Cx
        coup_im = K * Cy

        # combine
        dxk = growth_re + nonlin_re + coup_re
        dyk = growth_im + nonlin_im + coup_im

        du[xk_idx] = dxk
        du[yk_idx] = dyk
    end
end

"""
    gene_osc_ode_level!(du, u, p_scale, offset, t)

A simpler amplitude-phase or 1D gene model. We'll do amplitude-phase:
   Let Gk = (rk, θk) for each gene oscillator k.

p_scale might have:
  :μg, :ωg => param arrays
  :Kg => coupling, :A => adjacency
Or define your own simpler approach if you like.
"""
function gene_osc_ode_level!(du, u, p_scale, offset, t)
    μg  = p_scale.μg
    ωg  = p_scale.ωg
    Kg  = p_scale.Kg
    A   = p_scale.A

    n_g = length(ωg)
    for k in 1:n_g
        rk_idx = offset + 2(k-1) + 1
        th_idx = offset + 2(k-1) + 2
        rk = u[rk_idx]
        θk = u[th_idx]

        # For amplitude-phase oscillator:
        #   dr/dt = μg(k)*r_k - r_k^3 + coupling ...
        #   dθ/dt = ωg(k) + ...
        # plus adjacency-based coupling for amplitude or phase
        dr = μg[k]*rk - rk^3
        dθ = ωg[k]

        # adjacency-based coupling example
        for j in 1:n_g
            if j != k && A[k,j] != 0.0
                # find r_j, θ_j
                rj_idx = offset + 2(j-1) + 1
                thj_idx = offset + 2(j-1) + 2
                rj = u[rj_idx]
                θj = u[thj_idx]
                # some simplest coupling
                dr += Kg*A[k,j]*(rj - rk)
                dθ += Kg*A[k,j]*sin(θj - θk)
            end
        end

        du[rk_idx] = dr
        du[th_idx] = dθ
    end
end

###############################################################################
# 2) Master ODE That Calls Sub-Levels + Cross-Level Coupling
###############################################################################

"""
    multiscale_ode!(du, u, p, t)

Main ODE that:
  1) Calls each scale's sub-level ODE
  2) Incorporates cross-level couplings if needed

We rely on p.scales to define:
   p.scales = [
      (name="behavior", model_type="HKB", size=n_b, offset=offset_b, p_scale=NamedTuple(...)),
      (name="neural",   model_type="SL",  size=2n_n, offset=offset_n, p_scale=NamedTuple(...)),
      (name="genes",    model_type="GENE",size=2n_g, offset=offset_g, p_scale=NamedTuple(...))
   ]

Then we have p.cross_level = function(du, u, scales, t) that modifies du 
with cross-level interactions (top-down/bottom-up).
"""
function multiscale_ode!(du, u, p, t)
    # zero out du
    fill!(du, 0.0)

    # 1) Each level’s internal ODE
    for scl in p.scales
        model_type = scl.model_type
        offset     = scl.offset
        p_scale    = scl.p_scale
        if model_type == "HKB"
            hkb_ode_level!(du, u, p_scale, offset, t)
        elseif model_type == "SL"
            sl_ode_level!(du, u, p_scale, offset, t)
        elseif model_type == "GENE"
            gene_osc_ode_level!(du, u, p_scale, offset, t)
        else
            # fallback or error
            error("Unknown model_type: $model_type")
        end
    end

    # 2) Cross-level couplings
    if haskey(p, :cross_level)
        p.cross_level(du, u, p.scales, t)
    end
end

###############################################################################
# 3) Compute Criticality Measures and Plot Features Over Time
###############################################################################

"""
    compute_rolling_branching_ratio(sol, scales; win_size=100)

Computes the branching ratio (λ) over rolling windows of size `win_size`.
"""
function compute_rolling_branching_ratio(sol, scales; win_size=50)
    t_vals = sol.t
    num_windows = length(t_vals) - win_size + 1
    λ_values = Dict(name => zeros(num_windows) for name in [s.name for s in scales])

    for scale in scales
        name, offset, ssize = scale.name, scale.offset, scale.size
        u_vals = hcat(sol.u...)

        for i in 1:num_windows
            window_activity = mean(abs.(u_vals[offset:offset+ssize-1, i:i+win_size-1]), dims=1)
            λ_values[name][i] = mean(window_activity[2:end] ./ window_activity[1:end-1])
        end
    end
    
    return λ_values, t_vals[1:num_windows]  # Return time-aligned rolling λ
end

"""
    compute_rolling_fluctuation_scaling(sol, scales; win_size=100)

Computes fluctuation scaling (β) in rolling windows.
"""
function compute_rolling_fluctuation_scaling(sol, scales; win_size=50)
    t_vals = sol.t
    num_windows = length(t_vals) - win_size + 1
    scaling_exponents = Dict(name => zeros(num_windows) for name in [s.name for s in scales])

    for scale in scales
        name, offset, ssize = scale.name, scale.offset, scale.size
        u_vals = hcat(sol.u...)

        for i in 1:num_windows
            window_data = abs.(u_vals[offset:offset+ssize-1, i:i+win_size-1])
            means = mean(window_data, dims=2)[:, 1]  # Convert to 1D
            variances = var(window_data, dims=2)[:, 1]

            log_means = log.(means .+ eps())  # Avoid log(0)
            log_variances = log.(variances .+ eps())

            if length(log_means) > 1  # Ensure we have enough data points
                β = sum((log_means .- mean(log_means)) .* (log_variances .- mean(log_variances))) /
                    sum((log_means .- mean(log_means)).^2)
            else
                β = NaN  # Not enough data for regression
            end

            scaling_exponents[name][i] = β
        end
    end
    
    return scaling_exponents, t_vals[1:num_windows]  # Return time-aligned rolling β
end

"""
    compute_psd(sol, scales; fs=1.0)

Computes the Power Spectral Density (PSD) for each oscillator level.
"""
function compute_psd(sol, scales; fs=1.0)
    psd_values = Dict()
    freqs_dict = Dict()

    for scale in scales
        name, offset, ssize = scale.name, scale.offset, scale.size
        u_vals = hcat(sol.u...)  # Convert solution to matrix
        t_vals = sol.t  # Time points

        # Adjust size for SL and AP since they use 2 params per oscillator
        if name == "neural" || name == "genes"
            ssize = Int(ssize / 2)
        end

        # Extract time-series data for this level
        signals = u_vals[offset:offset+ssize-1, :]
        # signals = vcat(signals...)

        # Compute PSD using FFT
        psd_matrix = []
        # freqs = fftfreq(size(signals, 2), fs)
        freqs = 1:fs:size(signals, 2)

        for i in 1:ssize
            signal_fft = fft(signals[i, :])
            psd = abs2.(signal_fft) / length(signal_fft)
            push!(psd_matrix, psd[1:div(end, 2)])  # Keep positive frequencies only
        end

        # Store results
        psd_values[name] = reduce(hcat, psd_matrix)
        freqs_dict[name] = freqs[1:div(end, 2)]
    end

    return psd_values, freqs_dict
end

###############################################################################
# 3.5) Plot Criticality Features Over Time
###############################################################################

"""
    plot_criticality_measures(sol, scales)

Plots the evolution of criticality measures over time.
"""
function plot_criticality_measures(sol, scales)
    # t_vals = sol.t  # Time points
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=true)
    
    # Compute and extract values dynamically
    λ_values, _ = compute_rolling_branching_ratio(sol, scales)
    scaling_exponents, _ = compute_rolling_fluctuation_scaling(sol, scales)
    
    # Plot branching ratio evolution
    for scale in scales
        name = scale.name
        axs[2].plot(λ_values[name], label=name)
    end
    axs[2].set_ylabel("Branching Ratio (λ)")
    axs[2].legend()
    axs[2].set_title("Branching Ratio Over Time")
    
    # Plot fluctuation scaling exponent evolution
    for scale in scales
        name = scale.name
        axs[1].plot(scaling_exponents[name], label=name)
    end
    axs[1].set_ylabel("Fluctuation Scaling (β)")
    axs[1].legend()
    axs[1].set_title("Fluctuation Scaling Exponent Over Time")
    
    axs[2].set_xlabel("Time")
    plt.tight_layout()
    # plt.show()
    plt.savefig("out/criticality_measures.png")
end

"""
    plot_psd_and_criticality(sol, scales)

Plots both criticality measures and frequency analysis.
"""
function plot_psd_and_criticality(sol, scales)
    t_vals = sol.t  # Time points
    
    # fig = GLMakie.Figure(size=(1000, 800))
    # axs = [Axis(fig[i, 1], title="Branching Ratio") for i in 1:3]

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=true)
    
    # Compute criticality measures
    λ_values, _ = compute_rolling_branching_ratio(sol, scales)
    scaling_exponents, _ = compute_rolling_fluctuation_scaling(sol, scales)

    # Compute PSD
    psd_values, freqs_dict = compute_psd(sol, scales)

    # # Plot Branching Ratio
    # for (name, λ) in λ_values
    #     lines!(axs[1], t_vals[2:end], fill(λ, length(t_vals)-1), label=name)
    # end
    # axs[1].ylabel = "Branching Ratio (λ)"

    # # Plot Fluctuation Scaling
    # for (name, β) in scaling_exponents
    #     lines!(axs[2], t_vals, fill(β, length(t_vals)), label=name)
    # end
    # axs[2].ylabel = "Fluctuation Scaling (β)"

    # Plot branching ratio evolution
    for scale in scales
        name = scale.name
        axs[2].plot(λ_values[name], label=name)
    end
    axs[2].set_ylabel("Branching Ratio (λ)")
    axs[2].legend()
    axs[2].set_title("Branching Ratio Over Time")
    
    # Plot fluctuation scaling exponent evolution
    for scale in scales
        name = scale.name
        axs[1].plot(scaling_exponents[name], label=name)
    end
    axs[1].set_ylabel("Fluctuation Scaling (β)")
    axs[1].legend()
    axs[1].set_title("Fluctuation Scaling Exponent Over Time")
    
    # plt.show()
    # plt.savefig("out/criticality_measures.png")

    # Plot PSDs
    # axs_psd = Axis(fig[3, 1], title="Power Spectral Density (PSD)")
    # for (name, psd_matrix) in psd_values
    #     freqs = freqs_dict[name]
    #     mean_psd = mean(psd_matrix, dims=2)
    #     lines!(axs_psd, freqs, mean_psd, label=name)
    # end
    # axs_psd.ylabel = "PSD Power"
    # axs_psd.xlabel = "Frequency (Hz)"

    for scale in scales
        name = scale.name
        mean_psd = mean(psd_values[name], dims=2)
        # mean_psd = mean_psd[:100]
        axs[3].plot(mean_psd, label=name)
    end
    axs[3].set_ylabel("PSD Power")
    axs[3].legend()
    axs[3].set_title("Power Spectrum Density")
    axs[3].set_xlabel("Frequency (Hz)")

    # Show legend
    # fig[3, 1] = Legend(fig, axs_psd, "Levels", framevisible=false)
    
    axs[2].set_xlabel("Time")
    plt.tight_layout()
    # display(fig)
    plt.savefig("out/psd_criticality_measures.png")
end

###############################################################################
# 4) Generate 3D Movies of System Dynamics
###############################################################################

# using GLMakie, FileIO, Colors, GeometryBasics

"""
    generate_3D_movie(sol, scales, filename="multiscale_dynamics.mp4")

Creates an animated 3D scatter plot of oscillator dynamics with proper layering and phase encoding.
"""
function generate_3D_movie(sol, scales, filename="out/multiscale_dynamics.mp4")
    scatter_points = Observable(Vector{Point3f}())
    # time = Observable(0)
    scatter_colors = Observable(Vector{RGBAf}())

    # scatter_points = Observable(Point3f[(1,1,1)])
    # scatter_colors = Observable([RGBAf(0.5,0.5,0.5,1.0)])

    # Create initial scatter plot
    # scatter!(ax, scatter_points, color=scatter_colors, markersize=15)
    # scatter!(ax, scatter_points, markersize=15)
    
    # fig = GLMakie.Figure(size=(900, 650))
    # ax = Axis3(fig[1, 1], title="3D Dynamics of Multiscale System",
            #    xlabel="X", ylabel="Y", zlabel="Z")

    # fig, ax, scat = GLMakie.scatter(scatter_points, color=scatter_colors, axis=(;type=Axis3), markersize=15)
    # fig, ax, scat = GLMakie.scatter(scatter_points, axis=(;type=Axis3, title=@lift("t = $time")), markersize=15)
    # fig, ax, scat = GLMakie.scatter(scatter_points, axis=(;type=Axis3), markersize=15)

    u_vals = hcat(sol.u...)
    t_vals = sol.t
    num_frames = length(t_vals)

    # Define base Z positions for layers
    z_positions = Dict(
        "behavior" => 3.0,
        "neural" => 2.0,
        "genes" => 1.0
    )

    offsets = Dict(
        "behavior" => Dict(k => (cos(2π * k / scales[1].size), sin(2π * k / scales[1].size)) for k in 1:scales[1].size),  # Circular layout
        "neural"   => Dict(k => ((k-1) % 2 - 0.5, (k-1) ÷ 2 - 0.5) for k in 1:Int(scales[2].size / 2)),  # Grid layout
        "genes"    => Dict(k => (k-1, 0) for k in 1:Int(scales[3].size / 2))  # Linear layout
    )

    # Assign base X-Y offsets to prevent overlap
    x_offsets = Dict(scale.name => collect(range(-1, stop=1, length=scale.size)) for scale in scales)
    y_offsets = Dict(scale.name => collect(range(-1, stop=1, length=scale.size)) for scale in scales)

    level_colors = Dict(
        "behavior" => RGBAf(0.0, 0.0, 1.0, 1.0),  # Blue
        "neural"   => RGBAf(1.0, 0.5, 0.0, 1.0),  # Orange
        "genes"    => RGBAf(0.0, 1.0, 0.0, 1.0)   # Green
    )

    # scatter_colors = []
    # for scale in scales
    #     name, offset, size = scale.name, scale.offset, scale.size
    #     if name == "neural" || name == "genes"
    #         size = Int(size / 2)  # Adjust for two-parameter oscillators
    #     end
    #     append!(scatter_colors, fill(level_colors[name], size))
    # end
    # scatter_colors_obs = Observable(scatter_colors)

    # println(scatter_colors_obs)
    # println(size(scatter_colors_obs))

    fig, ax, scat = GLMakie.scatter(scatter_points, color=scatter_colors, axis=(;type=Axis3), markersize=15)
    GLMakie.limits!(ax, -10, 10, -10, 10, 0, 4)

    # Animation loop
    record(fig, filename, 1:num_frames; framerate=30) do frame
        # Create new lists of points and colors for the current frame
        new_points = Point3f[]
        new_colors = RGBAf[]

        for scale in scales
            scale_osc_offset_x = 0
            scale_osc_offset_y = 0
            name, offset, ssize = scale.name, scale.offset, scale.size

            # println(name, offset, size)

            if name == "neural" || name == "genes"
                ssize = Int(ssize / 2)     # two time the number of params
            end

            for k in 1:ssize
                x_offset, y_offset = offsets[name][k]
                if name == "behavior"  # HKB: Phase to XY
                    ϕ = u_vals[offset + (k-1), frame]
                    x, y = cos(ϕ), sin(ϕ)
                    z = 3.0

                elseif name == "neural"  # SL: Real & Imaginary Components
                    x = u_vals[offset + 2(k-1) + 1, frame]
                    y = u_vals[offset + 2(k-1) + 2, frame]
                    z = 2.0

                elseif name == "genes"  # AP: Amplitude & Phase
                    r = u_vals[offset + 2(k-1) + 1, frame]
                    θ = u_vals[offset + 2(k-1) + 2, frame]
                    x, y = r * cos(θ), r * sin(θ)
                    z = 1.0
                end

                # z = zs[k]  # Assign unique Z height

                # Compute phase color encoding
                # phase = atan(y, x)
                # color = RGBAf(sin(phase), cos(phase), 0.5, 1.0)

                # push!(new_points, Point3f(x, y, z))
                # push!(new_colors, color)

                append!(new_points, [Point3f(x+x_offset, y+y_offset, z)])
                append!(new_colors, [level_colors[name]])
            end

            # x = u_vals[offset, frame] .+ x_offsets[name]
            # y = u_vals[offset+1, frame] .+ y_offsets[name]
            # z = fill(z_positions[name], size)



            # # Compute phase color encoding
            # # phase = atan.(u_vals[offset+1, frame], u_vals[offset, frame])
            # # colors = [RGBAf(sin(p), cos(p), 0.5, 1.0) for p in phase]

            # # Store points and colors for this scale
            # append!(new_points, [Point3f(cos(x[i]), sin(y[i]), z[i]) for i in 1:size])
            # # append!(new_colors, colors)
        end

        # println(frame)

        # println("scatter points: ", scatter_points)
        # println("scatter colors", scatter_colors)

        # println("new points:", new_points)
        # println("new colors:", new_colors)

        # Overwrite Observables with new data for this frame
        # scatter_points[] = push!(scatter_points[], new_points)
        # scatter_colors[] = push!(scatter_colors[], new_colors)

        scatter_points[] = new_points
        scatter_colors[] = new_colors
        # time[] = frame
    end

    println("3D movie saved as $filename")
    return filename
end

# ###############################################################################
# # 5) Function to Plot System Dynamics
# ###############################################################################

# """
#     plot_system_dynamics(sol, scales)

# Plots the system dynamics for each level: Behavior (HKB), Neural (SL), and Genes (GENE).
# """
# function plot_system_dynamics(sol, scales)
#     t_vals = sol.t  # Time points

#     # Initialize figure with subplots for each level
#     fig, axs = plt.subplots(length(scales), 1, figsize=(10, 8), sharex=true)

#     for (idx, scale) in enumerate(scales)
#         name = scale.name
#         offset = scale.offset
#         size = scale.size
#         model_type = scale.model_type

#         # Extract the corresponding variables from the solution
#         u_vals = hcat(sol.u...)  # Convert list of vectors into a matrix
#         state_data = u_vals[offset : offset + size - 1, :]  # Extract relevant rows

#         ax = axs[idx]
#         ax.plot(t_vals, state_data', alpha=0.8)

#         ax.set_ylabel("$name\n($model_type)")
#         ax.legend(["Var $i" for i in 1:size], loc="upper right", fontsize=8)

#     end

#     axs[end].set_xlabel("Time")
#     plt.suptitle("Multiscale System Dynamics", fontsize=14)
#     plt.tight_layout()
#     plt.show()

# end

###############################################################################
# 6) Example Usage
###############################################################################

function main()
    rng = MersenneTwister(1234)

    # Suppose:
    # Level1: Behavior (HKB) with n_b=3 phases
    # Level2: Neural (SL) with n_n=2 oscillators => total dimension=2*2=4
    # Level3: Genes (GENE) with n_g=2 amplitude-phase => dimension=4

    n_b = 5
    n_n = 8
    n_g = 5

    # We'll define offsets in the big state vector:
    offset_b = 1
    offset_n = offset_b + n_b
    offset_g = offset_n + 2 * n_n
    dim_total = offset_g + 2 * n_g

    # 1) Behavior scale: HKB
    # adjacency for HKB: simple all-to-all minus identity
    # => bilateral graph of tentacles + hub
    A_b = ones(n_b,n_b) .- I(n_b)
    p_behavior = (
        # ω = [1.0, 1.05, 0.95], # intrinsic freq
        ω = rand(n_b),
        a = 0.5,
        b = -0.3,
        A = A_b
    )

    println("behavior connectivity: ", A_b)

    # 2) Neural scale: SL
    # adjacency for neural: fully connected or ring
    # => get small world connectivity
    A_n = ones(n_n,n_n) .- I(n_n)
    p_neural = (
        μ = 0.1,
        # ω = [1.0, 1.2],
        ω = rand(n_n),
        K = 0.4,
        A = A_n
    )

    println("neaural connectivity: ", A_n)

    # 3) Gene scale: GENE
    # adjacency for gene
    # => directed graph of 
    A_g = ones(n_g,n_g) .- I(n_g)
    p_genes = (
        # μg = [0.05, 0.07],
        μg = rand(n_g),
        # ωg = [1.0, 1.1],
        ωg = rand(n_g),
        Kg = 0.2,
        A  = A_g
    )

    println("gene connectivity: ", A_g)

    # Put scale definitions together
    # Each scale has (name, model_type, size, offset, p_scale)
    scales = [
        (name="behavior", model_type="HKB",  size=n_b,   offset=offset_b, p_scale=p_behavior),
        (name="neural",   model_type="SL",   size=2*n_n, offset=offset_n, p_scale=p_neural),
        (name="genes",    model_type="GENE", size=2*n_g, offset=offset_g, p_scale=p_genes)
    ]

    # Example cross-level coupling function
    cross_level_func = function(du, u, scales, t)
        # This is where you handle top-down (behavior->neural, neural->genes, etc.)
        # or bottom-up interactions. For demonstration, let's do something small:
        # E.g. let the average behavior phase shift neural's μ param, or something.
        # We'll keep it simple: if the behavior average phase is large, add some small
        # term to du in the neural part. This is fully up to your imagination.

        # 1) find the behavior scale (HKB) offset
        sc_behav = filter(s->s.name=="behavior", scales)[1]
        sc_neur  = filter(s->s.name=="neural", scales)[1]
        n_b = sc_behav.size
        offset_b = sc_behav.offset
        offset_n = sc_neur.offset

        # average behavior phase
        mean_phase = 0.0
        for i in 1:n_b
            mean_phase += u[offset_b + (i-1)]
        end
        mean_phase /= n_b

        # Let's just add a small top-down push to neural x-states
        n_n2 = sc_neur.size ÷ 2
        for j in 1:n_n2
            xj_idx = offset_n + 2*(j-1) + 1
            # du[xj_idx] += 0.01*(mean_phase) # example
            du[xj_idx] += 0.01 * sin(mean_phase)
        end

        # Similarly, do a bottom-up effect from neural amplitude to the genes
        sc_gen = filter(s->s.name=="genes", scales)[1]
        offset_g = sc_gen.offset
        n_g2 = sc_gen.size ÷ 2

        # measure neural amplitude: sum over SL oscillators
        amp_sum = 0.0
        for j in 1:n_n2
            xj_idx = offset_n + 2*(j-1) + 1
            yj_idx = offset_n + 2*(j-1) + 2
            xj = u[xj_idx]
            yj = u[yj_idx]
            amp_sum += sqrt(xj^2 + yj^2)
        end
        amp_mean = amp_sum / n_n2

        # let that modulate the gene's dr/dt
        for g in 1:n_g2
            r_g_idx = offset_g + 2*(g-1) + 1
            du[r_g_idx] += 0.02*amp_mean
        end
    end

    # Parameter container for the ODE
    p = (
        scales=scales,
        cross_level=cross_level_func
    )

    # Build initial condition
    u0 = [0.01*(rand(rng)-0.5) for _ in 1:dim_total]

    # ODE problem
    tspan = (0.0, 200.0)
    prob = ODEProblem(multiscale_ode!, u0, tspan, p)

    sol = solve(prob, Tsit5(); dt=0.1, save_everystep=true)

    println("Solution done. Let's do a quick summary of final states:")
    println(sol[end])

    # # Compute criticality measures
    # compute_branching_ratio(sol, scales)
    # compute_fluctuation_scaling(sol, scales)
    
    # # Generate 3D movie
    # generate_3D_movie(sol, scales, "out/multiscale_dynamics.png")

    return sol, scales
end

end # module

using .MultiscaleFramework: main, generate_3D_movie, plot_psd_and_criticality
@time sol, scales = main()

# println(sol)
println(size(sol))

# Compute criticality measures
# compute_rolling_branching_ratio(sol, scales, win_size=100)
# compute_rolling_fluctuation_scaling(sol, scales, win_size=100)

# λ_series, t_λ = compute_rolling_branching_ratio(sol, scales, win_size=100)
# β_series, t_β = compute_rolling_fluctuation_scaling(sol, scales, win_size=100)


# plot_criticality_measures(sol, scales)
plot_psd_and_criticality(sol, scales)
generate_3D_movie(sol, scales)