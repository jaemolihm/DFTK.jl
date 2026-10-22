# This file contains functions to handle the symetries.
# The type SymOp is defined in Symop.jl

include("external/spglib.jl")

# A symmetry (W, w) (or (S, τ)) induces a symmetry in the Brillouin zone that the
# Hamiltonian at S k is unitary equivalent to that at k, which we exploit to reduce
# computations. The relationship is
#   S = W'
#   τ = -W^-1 w
# (valid both in reduced and Cartesian coordinates). In our notation the rotation matrix
# W and translation w are such that, for each atom of type A at position a, W a + w is also
# an atom of type A.

# The full (reducible) Brillouin zone is implicitly represented by a set of (irreducible)
# kpoints (see explanation in docs). Each irreducible k-point k comes with a list of
# symmetry operations (S, τ) (containing at least the trivial operation (I, 0)), where S is
# a unitary matrix (/!\ in Cartesian but not in reduced coordinates) and τ a translation
# vector. The k-point is then used to represent implicitly the information at all the
# kpoints Sk. The relationship between the Hamiltonians is
#   H_{Sk} = U H_k U*, with
#   (Uu)(x) = u(W x + w)
# or in Fourier space
#   (Uu)(G) = e^{-i G τ} u(S^-1 G)
# In particular, we can choose the eigenvectors at Sk as u_{Sk} = U u_k

# We represent then the BZ as a set of irreducible points `kpoints`, and a set of weights
# `kweights` (summing to 1). The value of observables is given by a weighted sum over the
# irreducible kpoints, plus a symmetrization operation (which depends on the particular way
# the observable transforms under the symmetry).

# There is by decreasing cardinality
# - The group of symmetry operations of the lattice
# - The group of symmetry operations of the crystal (model.symmetries)
# - The group of symmetry operations of the crystal that preserves the BZ mesh (basis.symmetries)

# See https://juliamolsim.github.io/DFTK.jl/stable/developer/symmetries/ for details.

"""
`:none` if no element has a magnetic moment, else `:collinear` or `:full`.
"""
function determine_spin_polarization(magnetic_moments)
    isempty(magnetic_moments) && return :none
    all_magmoms = normalize_magnetic_moment.(magnetic_moments)
    all(iszero, all_magmoms) && return :none
    all(iszero(magmom[1:2]) for magmom in all_magmoms) && return :collinear

    :full
end


"""
Return the Symmetry operations given a `hall_number`.

This function allows to directly access to the space group operations in the
`spglib` database. To specify the space group type with a specific choice,
`hall_number` is used.

The definition of `hall_number` is found at
[Space group type](https://spglib.readthedocs.io/en/latest/dataset.html#dataset-spg-get-dataset-spacegroup-type).
"""
function symmetry_operations(hall_number::Integer)
    Ws, ws = Spglib.get_symmetry_from_database(hall_number)
    [SymOp(W, w) for (W, w) in zip(Ws, ws)]
end

@doc raw"""
Return the symmetries given an atomic structure with optionally designated magnetic moments
on each of the atoms. The symmetries are determined using spglib.
"""
@timing function symmetry_operations(lattice, atoms, positions, magnetic_moments=[];
                                     tol_symmetry=SYMMETRY_TOLERANCE,
                                     check_symmetry=SYMMETRY_CHECK)
    spin_polarization = determine_spin_polarization(magnetic_moments)
    dimension   = count(!iszero, eachcol(lattice))
    if isempty(atoms) || dimension != 3
        # spglib doesn't support these cases, so we default to no symmetries
        return [one(SymOp)]
    end

    if spin_polarization == :full
        @warn("Symmetry detection not yet supported in full spin polarization. " *
              "Returning no symmetries")
        return [one(SymOp)]
    end

    atom_groups = [findall(Ref(pot) .== atoms) for pot in Set(atoms)]
    cell = spglib_cell(lattice, atom_groups, positions, magnetic_moments)
    Ws, ws = try
        if spin_polarization == :none
            Spglib.get_symmetry(cell, tol_symmetry)
        elseif spin_polarization == :collinear
            rotations, translations, spin_flips = Spglib.get_symmetry_with_site_tensors(
                cell, tol_symmetry)
            # Keep only the symmetries that don't flip the spins
            rotations[spin_flips.==1], translations[spin_flips.==1]
            # Taking into account the symmetries that flip spins here would mean detecting 
            # antiferromagnetic order (where the down orbitals are the up orbitals 
            # translated by half a cell) and optimizing by computing only the up channel of 
            # the orbitals and building the density by symmetry. So, the density has two 
            # components, but the orbitals have only one spin channel.
            # This would cut runtime by a factor 2.
        end
    catch e
        if e isa Spglib.SpglibError
            msg = ("spglib failed to get the symmetries. Check your lattice, use a " *
                   "uniform BZ mesh or disable symmetries. Spglib reported : " * e.msg)
            throw(Spglib.SpglibError(msg))
        else
            rethrow()
        end
    end

    symmetries = [SymOp(W, w) for (W, w) in zip(Ws, ws)]
    if check_symmetry
        _check_symmetries(symmetries, lattice, atom_groups, positions; tol_symmetry)
    end
    symmetries
end
function symmetry_operations(system::AbstractSystem; kwargs...)
    pseudopotentials = fill(nothing, length(system))
    parsed = parse_system(system, pseudopotentials)
    symmetry_operations(parsed.lattice, parsed.atoms, parsed.positions,
                        parsed.magnetic_moments; kwargs...)
end

@timing function _check_symmetries(symmetries::AbstractVector{<:SymOp},
                                   lattice, atom_groups, positions;
                                   tol_symmetry=SYMMETRY_TOLERANCE)
    # Check (W, w) maps atoms to equivalent atoms in the lattice
    for symop in symmetries
        W, w = symop.W, symop.w

        # Check (A W A^{-1}) is orthogonal
        Wcart = lattice * W / lattice
        if maximum(abs, Wcart'Wcart - I) > tol_symmetry
            error("Issue in symmetry determination: Non-orthogonal rotation matrix.")
        end

        for group in atom_groups
            group_positions = positions[group]
            for coord in group_positions
                # If all elements of a difference in diffs is integer, then
                # W * coord + w and pos are equivalent lattice positions
                if !any(c -> is_approx_integer(W * coord + w - c; atol=tol_symmetry), group_positions)
                    error("Issue in symmetry determination: Cannot map the atom at position " *
                          "$coord to another atom of the same element under the symmetry " *
                          "operation (W, w):\n($W, $w)")
                end
            end
        end
    end
end

# Approximate in; can be performance-critical, so we optimize in case of rationals
_is_approx_in(x::AbstractArray{<:Rational}, X)  = any(isequal(x), X)
_is_approx_in(x::AbstractArray{T}, X) where {T} = any(y -> isapprox(x, y; atol=sqrt(eps(T))), X)

"""
Filter out the symmetry operations that don't respect the symmetries of the discrete BZ grid
"""
function symmetries_preserving_kgrid(symmetries, kgrid::AbstractKgrid)
    # First apply symmetries as the provided k-points can be arbitrary
    # (e.g. only along a line or similar)
    all_kcoords = unfold_kcoords(reducible_kcoords(kgrid).kcoords, symmetries)
    kcoords_normalized = normalize_kpoint_coordinate.(all_kcoords)
    function preserves_grid(symop)
        all(_is_approx_in(normalize_kpoint_coordinate(symop.S * k), kcoords_normalized)
            for k in kcoords_normalized)
    end
    filter(preserves_grid, symmetries)
end
# Special case for Monkhorst-Pack grids since the generic function above
# is quadratic in the number of k-points.
function symmetries_preserving_kgrid(symmetries, kgrid::MonkhorstPack)
    # Spglib only supports shifts that are 0 or 1//2, so the same holds true
    # for our irreducible_kcoords function. We disable all symmetries in this case.
    are_shifts_supported_by_spglib = all(ks -> ks in (0, 1//2), kgrid.kshift)
    if !are_shifts_supported_by_spglib && length(symmetries) > 1
        return filter(isone, symmetries)
    end

    # we have to check if S * k is in the grid for all k
    # by linearity, it is enough to check the origin and
    # (1/kgrid_size[1], 0, 0), (0, 1/kgrid_size[2], 0), (0, 0, 1/kgrid_size[3]).
    kcoords_normalized = normalize_kpoint_coordinate.(reducible_kcoords(kgrid).kcoords)

    function preserves_kgrid(symop)
        function _check(dx, dy, dz)
            k = (kgrid.kshift .+ Vec3(dx, dy, dz)) .// kgrid.kgrid_size
            normalize_kpoint_coordinate(symop.S * k) ∈ kcoords_normalized
        end
        _check(0, 0, 0) && _check(1, 0, 0) && _check(0, 1, 0) && _check(0, 0, 1)
    end

    filter(preserves_kgrid, symmetries)
end

"""
Filter out the symmetry operations that don't respect the symmetries of the discrete real-space grid
"""
function symmetries_preserving_rgrid(symmetries, fft_size)
    is_in_grid(r) = all(zip(r, fft_size)) do (ri, size)
        abs(ri * size - round(ri * size)) / size ≤ SYMMETRY_TOLERANCE
    end

    onehot3(i) = (x = zeros(Bool, 3); x[i] = true; Vec3(x))
    function preserves_grid(symop)
        all(is_in_grid(symop.W * onehot3(i) .// fft_size[i] + symop.w) for i=1:3)
    end

    filter(preserves_grid, symmetries)
end

@doc raw"""
Apply various standardisations to a lattice and a list of atoms. It uses spglib to detect
symmetries (within `tol_symmetry`), then cleans up the lattice according to the symmetries
(unless `correct_symmetry` is `false`) and returns the resulting standard lattice
and atoms. If `primitive` is `true` (default) the primitive unit cell is returned, else
the conventional unit cell is returned.
"""
function standardize_atoms(lattice, atoms, positions, magnetic_moments=[]; kwargs...)
    @assert length(atoms) == length(positions)
    @assert isempty(magnetic_moments) || (length(atoms) == length(magnetic_moments))
    atom_groups = [findall(Ref(pot) .== atoms) for pot in Set(atoms)]
    ret = spglib_standardize_cell(lattice, atom_groups, positions, magnetic_moments; kwargs...)
    (; ret.lattice, atoms, ret.positions, ret.magnetic_moments)
end


"""
Apply a symmetry operation to eigenvectors `ψk` at a given `kpoint` to obtain an
equivalent point in [-0.5, 0.5)^3 and associated eigenvectors (expressed in the
basis of the new ``k``-point).
"""
function apply_symop(symop::SymOp, basis, kpoint, ψk::AbstractVecOrMat)
    S, τ = symop.S, symop.τ
    isone(symop) && return kpoint, ψk

    # Apply S and reduce coordinates to interval [-0.5, 0.5)
    # Doing this reduction is important because
    # of the finite kinetic energy basis cutoff
    @assert all(-0.5 .≤ kpoint.coordinate .< 0.5)
    Sk_raw = S * kpoint.coordinate
    Sk = normalize_kpoint_coordinate(Sk_raw)
    kshift = convert.(Int, Sk - Sk_raw)
    @assert all(-0.5 .≤ Sk .< 0.5)

    # Check whether the resulting k-point is in the basis:
    ikfull = findfirst(1:length(basis.kpoints)) do idx
        all(isinteger, basis.kpoints[idx].coordinate - Sk)
    end
    if isnothing(ikfull)
        # Build new k-point datastructure
        Skpoint = Kpoint(basis, Sk, kpoint.spin)
    else
        Skpoint = basis.kpoints[ikfull]
        @assert Skpoint.coordinate ≈ Sk
    end

    # Since the eigenfunctions of the Hamiltonian at k and Sk satisfy
    #      u_{Sk}(x) = u_{k}(S^{-1} (x - τ))
    # their Fourier transform is related as
    #      ̂u_{Sk}(G) = e^{-i G \cdot τ} ̂u_k(S^{-1} G)
    invS = Mat3{Int}(inv(S))
    Gs_full = [G + kshift for G in G_vectors(basis, Skpoint)]
    ψSk = zero(ψk)
    for iband = 1:size(ψk, 2)
        for (ig, G_full) in enumerate(Gs_full)
            igired = index_G_vectors(basis, kpoint, invS * G_full)
            @assert igired !== nothing
            ψSk[ig, iband] = cis2pi(-dot(G_full, τ)) * ψk[igired, iband]
        end
    end

    Skpoint, ψSk
end

"""
Apply a symmetry operation to a density.
"""
function apply_symop(symop::SymOp, basis, ρin; kwargs...)
    isone(symop) && return ρin
    symmetrize_ρ(basis, ρin; symmetries=[symop], kwargs...)
end

# Accumulates the symmetrized versions of the density ρin into ρaccu (in Fourier space).
# No normalization is performed. This function is optimized for CPU and GPU.
function accumulate_over_symmetries!(ρaccu, ρin, basis::PlaneWaveBasis{T}, symmetries) where {T}
    # For each G vector and symmetry S:
    # Transform ρin -> to the partial density at S * k.
    #
    # Since the eigenfunctions of the Hamiltonian at k and Sk satisfy
    #      u_{Sk}(x) = u_{k}(S^{-1} (x - τ))
    # with Fourier transform
    #      ̂u_{Sk}(G) = e^{-i G \cdot τ} ̂u_k(S^{-1} G)
    # equivalently
    #     ρ ̂_{Sk}(G) = e^{-i G \cdot τ} ̂ρ_k(S^{-1} G) 
    if all(isone, symmetries)
        ρaccu .= ρin
        return ρaccu
    end

    Gs = reshape(G_vectors(basis), size(ρaccu))
    fft_size = basis.fft_size

    # Need to transfer  symmetry data to the GPU as isbit data
    symm_invS = to_device(basis.architecture, [Mat3{Int}(inv(symop.S)) for symop in symmetries])
    symm_τ = to_device(basis.architecture, [symop.τ for symop in symmetries])
    n_symm = length(symmetries)

    # Looping over symmetries inside of map! on G vectors allow for a single GPU kernel launch
    map!(ρaccu, Gs) do G
        acc = zero(complex(T))
        # Explicit loop over indices because AMDGPU does not support zip() in map!
        for i_symm in 1:n_symm
            invS = symm_invS[i_symm]
            τ = symm_τ[i_symm]
            idx = index_G_vectors(fft_size, invS * G)
            acc += isnothing(idx) ? zero(complex(T)) :
                        iszero(τ) ? ρin[idx] : cis2pi(-T(dot(G, τ))) * ρin[idx]
        end
        acc
    end
    ρaccu
end

"""
Symmetrize a density by applying all the basis (by default) symmetries and forming the average.
"""
@views @timing function symmetrize_ρ(basis, ρ::AbstractArray{T};
                                     symmetries=basis.symmetries) where {T}
    ρin_fourier  = fft(basis, ρ)
    ρout_fourier = zero(ρin_fourier)
    for σ = 1:size(ρ, 4)
        accumulate_over_symmetries!(ρout_fourier[:, :, :, σ],
                                    ρin_fourier[:, :, :, σ], basis, symmetries)
    end
    inv_fft = T <: Real ? irfft : ifft
    inv_fft(basis, ρout_fourier ./ length(symmetries))
end

"""
Symmetrize the stress tensor, given as a Matrix in Cartesian coordinates
"""
function symmetrize_stresses(model::Model, stresses; symmetries)
    # see (A.28) of https://arxiv.org/pdf/0906.2569.pdf
    stresses_symmetrized = zero(stresses)
    for symop in symmetries
        W_cart = matrix_red_to_cart(model, symop.W)
        stresses_symmetrized += W_cart * stresses / W_cart
    end
    stresses_symmetrized /= length(symmetries)
    stresses_symmetrized
end
function symmetrize_stresses(basis::PlaneWaveBasis, stresses; symmetries=basis.symmetries)
    symmetrize_stresses(basis.model, stresses; symmetries)
end

"""
Find the symmetry preimage of `position`, returning the corresponding index in `positions_group`.
"""
function find_symmetry_preimage(positions_group, position, symop;
                                tol_symmetry=SYMMETRY_TOLERANCE)
    # see (A.27) of https://arxiv.org/pdf/0906.2569.pdf
    # (but careful that our symmetries are r -> Wr+w, not R(r+f))
    other_at = symop.W \ (position - symop.w)

    # Find the index of the atom to which idx is mapped to by the symmetry operation.
    # To avoid issues due to numerical noise we compute the deviations from being
    # an integer shift (thus equivalent by translational symmetry) for all atoms in
    # the group and pick the smallest one.
    smallest_deviation, i_other_at = findmin(positions_group) do at
        δat = at - other_at
        maximum(abs, δat - round.(δat))
    end
    # Note, that without a fudging factor this occasionally fails:
    @assert smallest_deviation < 10tol_symmetry

    i_other_at
end

function symmetrize_forces(positions::AbstractVector, atom_groups::AbstractVector, forces;
                           symmetries, tol_symmetry=SYMMETRY_TOLERANCE)
    symmetrized_forces = zero(forces)
    for group in atom_groups, symop in symmetries
        positions_group = positions[group]
        for (idx, position) in enumerate(positions_group)
            i_other_at = find_symmetry_preimage(positions_group, position, symop; tol_symmetry)

            # (A.27) is in Cartesian coordinates, and since Wcart is orthogonal,
            # Fsymcart = Wcart * Fcart <=> Fsymred = inv(Wred') Fred
            symmetrized_forces[group[idx]] += inv(symop.W') * forces[group[i_other_at]]
        end
    end
    symmetrized_forces / length(symmetries)
end
function symmetrize_forces(model::Model, forces; kwargs...)
    symmetrize_forces(model.positions, model.atom_groups, forces; kwargs...)
end

"""
Symmetrize the forces in *reduced coordinates*, forces given as an array `forces[iel][α,i]`.
"""
function symmetrize_forces(basis::PlaneWaveBasis, forces)
    symmetrize_forces(basis.model, forces; basis.symmetries)
end

"""
Symmetrize the Hubbard occupation matrix according to the l quantum number of the manifold.
"""
function symmetrize_hubbard_n(model, manifold::ResolvedOrbitalManifold,
                              hubbard_n::Array{Matrix{Complex{T}}};
                              symmetries, tol_symmetry=SYMMETRY_TOLERANCE) where {T}
    # For now we apply symmetries only on nII terms, not on cross-atom terms (nIJ)
    # WARNING: To implement +V this will need to be changed!

    positions = model.positions[manifold.iatoms]
    nspins = size(hubbard_n, 1)
    natoms = size(hubbard_n, 2)
    ldim = 2*manifold.l+1

    # Initialize the hubbard_n matrix
    ns = [zeros(Complex{T}, ldim, ldim) for _ in 1:nspins, _ in 1:natoms, _ in 1:natoms]
    for symmetry in symmetries
        Wcart = model.lattice * symmetry.W * model.inv_lattice
        WigD = wigner_d_matrix(manifold.l, Wcart)
        for σ in 1:nspins, iatom in 1:natoms
            sym_atom = find_symmetry_preimage(positions, positions[iatom], symmetry;
                                              tol_symmetry)
            ns[σ, iatom, iatom] .+= WigD' * hubbard_n[σ, sym_atom, sym_atom] * WigD
        end
    end
    ns ./= length(symmetries)
    ns
end

""""
Convert a `basis` into one that doesn't use BZ symmetry.
This is mainly useful for debug purposes (e.g. in cases we don't want to
bother thinking about symmetries).
"""
function unfold_bz(basis::PlaneWaveBasis)
    if length(basis.symmetries) == 1
        return basis
    else
        # TODO This can be optimized much better by avoiding the recomputation
        #      of the terms wherever possible.
        use_symmetry_for_kpoint_reduction = false
        return PlaneWaveBasis(basis.model, basis.Ecut, basis.fft_size,
                              basis.variational, basis.kgrid,
                              basis.symmetries_respect_rgrid,
                              use_symmetry_for_kpoint_reduction,
                              basis.comm_kpts, basis.architecture)
    end
end

# Normalized k-point coordinate rounded to SYMMETRY_TOLERANCE, used as a key to identify
# k-points. If x and y are both close to a round value, round(x)===round(y), except at zero
# where 0.0 and -0.0 are considered different. Add 0.0 to make both -0.0 and 0.0 equal to 0.0
function rounded_kcoord(k)
    digits = ceil(Int, -log10(SYMMETRY_TOLERANCE))
    normalize_kpoint_coordinate(round.(k; digits) .+ 0.0)
end

"""
For every k-point of `basis_target.kcoords_global` the pair `(ik_irred, isym)`, such that
`basis_irred.symmetries[isym]` maps the irreducible k-point `basis_irred.kcoords_global[ik_irred]`
(`ik_irred ≤ basis_irred.n_irreducible_kpoints`) to it, modulo a reciprocal lattice vector.
The first match in the order of irreducible k-points, then symmetries, is taken. The result
does not depend on the k-point distribution over MPI ranks.
"""
function unfold_map(basis_irred::PlaneWaveBasis, basis_target::PlaneWaveBasis)
    source = Dict{Vec3{Float64}, Tuple{Int, Int}}()
    for ik_irred = 1:basis_irred.n_irreducible_kpoints,
            (isym, symop) in enumerate(basis_irred.symmetries)
        Sk = symop.S * basis_irred.kcoords_global[ik_irred]
        get!(source, rounded_kcoord(Sk), (ik_irred, isym))
    end
    map(basis_target.kcoords_global) do k
        get(source, rounded_kcoord(k)) do
            error("k-point $k is not in the star of an irreducible k-point")
        end
    end
end

"""
Unfold k-point data from `basis_irred` to the k-points of `basis_target`, which must lie in
the stars of the irreducible k-points. `data` holds ψ (`is_ψ = true`, possibly a subset of
the bands) or a per-band quantity (eigenvalues, occupations), with the same number of bands
at all k-points. Each target k-point receives the data of the source given by
[`unfold_map`](@ref), transformed by the symmetry operation in the case of ψ. Unless the
two bases are the same, the output arrays do not alias `data` or each other.
"""
function unfold_array(basis_irred, basis_target, data, is_ψ)
    basis_irred == basis_target && return data
    basis_irred.comm_kpts == basis_target.comm_kpts ||
        error("unfold_array needs both bases on the same comm_kpts")
    basis_irred.model.lattice == basis_target.model.lattice ||
        error("unfold_array needs bases with the same lattice")
    (basis_irred.fft_size == basis_target.fft_size && basis_irred.Ecut == basis_target.Ecut &&
     basis_irred.variational == basis_target.variational) ||
        error("unfold_array needs bases with the same Ecut, fft_size and variational")
    n_spin = basis_irred.model.n_spin_components
    n_spin == basis_target.model.n_spin_components ||
        error("unfold_array needs bases with the same number of spin components")
    all(k -> all(-0.5 .≤ k .< 0.5), basis_target.kcoords_global) ||
        error("unfold_array needs target k-points normalized to [-0.5, 0.5)")
    all(k -> all(-0.5 .≤ k .< 0.5),
        basis_irred.kcoords_global[1:basis_irred.n_irreducible_kpoints]) ||
        error("unfold_array needs irreducible k-points normalized to [-0.5, 0.5)")
    basis_irred.architecture isa CPU || error("unfold_array is implemented for the CPU only")

    comm = basis_irred.comm_kpts
    rank = MPI.Comm_rank(comm)
    n_bands = is_ψ ? size.(data, 2) : length.(data)
    mpi_min(minimum(n_bands), comm) == mpi_max(maximum(n_bands), comm) ||
        error("unfold_array needs the same number of bands at all k-points")
    n_bands = first(n_bands)

    # Global spin-doubled k-point index (σ - 1) n_kpt + ik, as in `krange_allprocs`.
    function owner_ranks(basis)
        owner = zeros(Int, n_spin * length(basis.kcoords_global))
        for (r, kranges) in enumerate(basis.krange_allprocs), krange in kranges
            owner[krange] .= r - 1
        end
        owner
    end
    owner_irred  = owner_ranks(basis_irred)
    owner_target = owner_ranks(basis_target)
    n_kpt_irred  = length(basis_irred.kcoords_global)
    n_kpt_target = length(basis_target.kcoords_global)
    kmap = unfold_map(basis_irred, basis_target)
    # Global index of the source of the target `it`, and the index of the symmetry operation
    function source(it)
        σ = (it - 1) ÷ n_kpt_target + 1
        ik_irred, isym = kmap[it - (σ - 1) * n_kpt_target]
        (σ - 1) * n_kpt_irred + ik_irred, isym
    end
    local_irred(is) = findfirst(==(is), basis_irred.krange_thisproc_allspin)

    # The exchange plan is derived on every rank from the replicated map. Sends and receives
    # are posted in the order of the source index on both sides, so the messages of one
    # (sender, receiver) pair match in order with a single tag.
    sends = sort!(unique!([(first(source(it)), owner_target[it])
                           for it in eachindex(owner_target)
                           if owner_irred[first(source(it))] == rank && owner_target[it] != rank]))
    recvs = sort!(unique!([(first(source(it)), owner_irred[first(source(it))])
                           for it in basis_target.krange_thisproc_allspin
                           if owner_irred[first(source(it))] != rank]))

    # Source k-points rebuilt on basis_target: they set the G ordering of ψ for apply_symop
    function source_kpoint(is)
        σ = (is - 1) ÷ n_kpt_irred + 1
        Kpoint(basis_target, basis_irred.kcoords_global[is - (σ - 1) * n_kpt_irred], σ)
    end
    kpoints_source = if is_ψ
        sources_local = unique(first.(source.(basis_target.krange_thisproc_allspin)))
        Dict(is => source_kpoint(is) for is in sources_local)
    end

    # `data` may hold views, for which `similar` returns an Array
    received = Dict{Int, Array{eltype(first(data))}}()
    requests = MPI.Request[]
    for (is, dest) in sends
        push!(requests, MPI.Isend(data[local_irred(is)], comm; dest, tag=0))
    end
    for (is, src) in recvs
        size_source = is_ψ ? (length(G_vectors(basis_target, kpoints_source[is])), n_bands) :
                             (n_bands, )
        received[is] = similar(first(data), size_source)
        push!(requests, MPI.Irecv!(received[is], comm; source=src, tag=0))
    end
    MPI.Waitall(requests)

    map(basis_target.krange_thisproc_allspin, basis_target.kpoints) do it, kpt
        is, isym = source(it)
        is_local = owner_irred[is] == rank
        data_source = is_local ? data[local_irred(is)] : received[is]
        is_ψ || return copy(data_source)
        Skpoint, ψSk = apply_symop(basis_irred.symmetries[isym], basis_target,
                                   kpoints_source[is], data_source)
        # For the identity, apply_symop returns ψ on the G ordering of the source k-point
        @assert Skpoint.coordinate ≈ kpt.coordinate
        @assert size(ψSk, 1) == length(G_vectors(basis_target, kpt))
        is_local && ψSk === data_source ? copy(ψSk) : ψSk
    end
end

function unfold_bz(scfres)
    basis_unfolded = unfold_bz(scfres.basis)
    ψ = unfold_array(scfres.basis, basis_unfolded, scfres.ψ, true)
    eigenvalues = unfold_array(scfres.basis, basis_unfolded, scfres.eigenvalues, false)
    occupation = unfold_array(scfres.basis, basis_unfolded, scfres.occupation, false)
    energies, ham = energy_hamiltonian(basis_unfolded, ψ, occupation;
                                       scfres.ρ, scfres.hubbard_n, eigenvalues, scfres.εF)
    @assert energies.total ≈ scfres.energies.total
    new_scfres = (; basis=basis_unfolded, ψ, ham, eigenvalues, occupation)
    merge(scfres, new_scfres)
end

function unfold_kcoords(kcoords, symmetries)
    # unfold
    all_kcoords = [normalize_kpoint_coordinate(symop.S * kcoord)
                   for kcoord in kcoords, symop in symmetries]
    # uniquify
    unique(rounded_kcoord, all_kcoords)
end
