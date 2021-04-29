"""
Hartree term: for a decaying potential V the energy would be

1/2 ∫ρ(x)ρ(y)V(x-y) dxdy

with the integral on x in the unit cell and of y in the whole space.
For the Coulomb potential with periodic boundary conditions, this is rather

1/2 ∫ρ(x)ρ(y) G(x-y) dx dy

where G is the Green's function of the periodic Laplacian with zero
mean (-Δ G = sum_{R} 4π δ_R, integral of G zero on a unit cell).
"""
struct Hartree
    scaling_factor::Real  # to scale by an arbitrary factor (useful for exploration)
end
Hartree(; scaling_factor=1) = Hartree(scaling_factor)
(hartree::Hartree)(basis) = TermHartree(basis, hartree.scaling_factor)

struct TermHartree <: Term
    basis::PlaneWaveBasis
    scaling_factor::Real  # scaling factor, absorbed into poisson_green_coeffs
    # Fourier coefficients of the Green's function of the periodic Poisson equation
    poisson_green_coeffs
    inv_poisson_green_coeffs
end
function TermHartree(basis::PlaneWaveBasis{T}, scaling_factor) where T
    # Solving the Poisson equation ΔV = -4π ρ in Fourier space
    # is multiplying elementwise by 4π / |G|^2.
    poisson_green_coeffs = 4T(π) ./ [sum(abs2, basis.model.recip_lattice * G)
                                     for G in G_vectors(basis)]

    # Zero the DC component (i.e. assume a compensating charge background)
    poisson_green_coeffs[1] = 0
    poisson_green_coeffs = scaling_factor .* poisson_green_coeffs

    inv_poisson_green_coeffs = deepcopy(poisson_green_coeffs)
    for i in 1:length(inv_poisson_green_coeffs)
        if inv_poisson_green_coeffs[i] != 0.0
            inv_poisson_green_coeffs[i] = 1/inv_poisson_green_coeffs[i]
        end
    end

    TermHartree(basis, scaling_factor, poisson_green_coeffs,
               inv_poisson_green_coeffs)
end

@timing "ene_ops: hartree" function ene_ops(term::TermHartree, ψ, occ; ρ, kwargs...)
    basis = term.basis
    T = eltype(basis)
    ρtot = total_density(ρ)
    ρtot_fourier = r_to_G(basis, ρtot)
    pot_fourier = term.poisson_green_coeffs .* ρtot_fourier
    pot_real = G_to_r(basis, pot_fourier)
    E = real(dot(pot_fourier, ρtot_fourier) / 2)

    ops = [RealSpaceMultiplication(basis, kpoint, pot_real) for kpoint in basis.kpoints]
    (E=E, ops=ops)
end

function apply_kernel(term::TermHartree, dρ::RealFourierArray, dρspin; kwargs...)
    [from_fourier(dρ.basis, term.poisson_green_coeffs .* dρ.fourier)]
end

function apply_kernel_sqrt(term::TermHartree, dρ::RealFourierArray; kwargs...)
    from_fourier(dρ.basis, sqrt.(term.poisson_green_coeffs) .* dρ.fourier)
end

function apply_kernel_invsqrt(term::TermHartree, dρ::RealFourierArray; kwargs...)
    from_fourier(dρ.basis, sqrt.(term.inv_poisson_green_coeffs) .* dρ.fourier)
end


# for debugging

function compute_kernel(term::TermHartree; kwargs...)
    @assert term.basis.model.spin_polarization in (:none, :spinless, :collinear)
    vc_G = term.poisson_green_coeffs
    # Note that `real` here: if omitted, will result in high-frequency noise of even FFT grids
    K = real(G_to_r_matrix(term.basis) * Diagonal(vec(vc_G)) * r_to_G_matrix(term.basis))

    n_spin = term.basis.model.n_spin_components
    n_spin == 1 ? K : [K K; K K]
end

function apply_kernel(term::TermHartree, dρ; kwargs...)
    @assert term.basis.model.spin_polarization in (:none, :spinless, :collinear)
    dV = zero(dρ)
    dρtot = total_density(dρ)
    # note broadcast here: dV is 4D, and all its spin components get the same potential
    dV .= G_to_r(term.basis, term.poisson_green_coeffs .* r_to_G(term.basis, dρtot))
end

