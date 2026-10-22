@testmodule UnfoldBZ begin
using DFTK

function silicon_model(silicon, spin_polarization; symmetries=true)
    magnetic_moments = spin_polarization == :collinear ? [1.0, 1.0] : []
    model_DFT(silicon.lattice, silicon.atoms, silicon.positions; functionals=LDA(),
              temperature=0.01, magnetic_moments, symmetries)
end

# Basis without symmetries on the first `n_kpt` k-points of the full grid in
# `reducible_kcoords` order (all of them by default)
function explicit_full_basis(basis_irred, kgrid; comm_kpts, n_kpt=prod(kgrid))
    model = DFTK.Model(basis_irred.model; symmetries=false)
    kcoords = DFTK.reducible_kcoords(MonkhorstPack(kgrid)).kcoords[1:n_kpt]
    PlaneWaveBasis(model; basis_irred.Ecut, kgrid=ExplicitKpoints(kcoords),
                   basis_irred.fft_size, comm_kpts)
end

# Index of `basis.kpoints[ik]` in the spin-doubled global k-point list without duplicates
function global_index(basis, ik)
    kpt = basis.kpoints[ik]
    n_kpt = basis.n_irreducible_kpoints
    ik_global = findfirst(==(kpt.coordinate), basis.kcoords_global)
    (kpt.spin - 1) * n_kpt + ik_global
end
end


@testitem "unfold_array: unfolded ψ are eigenvectors" setup=[TestCases, UnfoldBZ] begin
    using DFTK
    using LinearAlgebra
    using .UnfoldBZ: silicon_model, explicit_full_basis
    using MPI
    silicon = TestCases.silicon

    Ecut  = 5
    kgrid = [3, 3, 3]
    tol   = 1e-10
    for spin_polarization in (:none, :collinear)
        model = silicon_model(silicon, spin_polarization)
        basis = PlaneWaveBasis(model; Ecut, kgrid)
        ρ = guess_density(basis, spin_polarization == :collinear ? [1.0, 1.0] : [])
        scfres = self_consistent_field(basis; tol, ρ)
        comm = basis.comm_kpts

        targets = [DFTK.unfold_bz(basis),
                   explicit_full_basis(basis, kgrid; comm_kpts=comm)]
        for basis_target in targets
            ψ   = DFTK.unfold_array(basis, basis_target, scfres.ψ, true)
            ε   = DFTK.unfold_array(basis, basis_target, scfres.eigenvalues, false)
            occ = DFTK.unfold_array(basis, basis_target, scfres.occupation, false)
            @test length(ψ) == length(ε) == length(occ) == length(basis_target.kpoints)

            # Residuals of the occupied bands of the Hamiltonian of the target basis
            (; ham) = energy_hamiltonian(basis_target, ψ, occ; scfres.ρ)
            residual = maximum(1:length(ψ)) do ik
                iocc = findall(occ[ik] .> 1e-10)
                Hψ = ham.blocks[ik] * ψ[ik][:, iocc]
                maximum(norm, eachcol(Hψ - ψ[ik][:, iocc] * Diagonal(ε[ik][iocc])))
            end
            @test DFTK.mpi_max(residual, comm) < 1e-6

            # The full-grid density equals the symmetric SCF density
            ρ_target = compute_density(basis_target, ψ, occ)
            @test norm(ρ_target - scfres.ρ) < 1e-8
            @test DFTK.weighted_ksum(basis_target, sum.(occ)) ≈ model.n_electrons
        end

        # Asserts the energy on the unfolded basis
        scfres_unfold = DFTK.unfold_bz(scfres)
        @test length(scfres_unfold.ψ) == length(scfres_unfold.basis.kpoints)

        if DFTK.mpi_nprocs(comm) > 1
            @test_throws ErrorException cell_to_supercell(scfres)
        end
    end
end

@testitem "unfold_array: independent of the number of MPI ranks" #=
    =#    setup=[TestCases, UnfoldBZ] begin
    using DFTK
    using MPI
    using Random
    using .UnfoldBZ: silicon_model, explicit_full_basis, global_index
    silicon = TestCases.silicon

    Ecut    = 5
    n_bands = 4
    for kgrid in ([2, 2, 2], [3, 3, 3]), spin_polarization in (:none, :collinear)
        model = silicon_model(silicon, spin_polarization)
        basis_mpi = PlaneWaveBasis(model; Ecut, kgrid)
        basis_ref = PlaneWaveBasis(model; Ecut, kgrid, comm_kpts=MPI.COMM_SELF)

        # Deterministic data from the global k-point index
        function random_data(basis)
            map(1:length(basis.kpoints)) do ik
                Random.seed!(global_index(basis, ik))
                ψk = randn(ComplexF64, length(G_vectors(basis, basis.kpoints[ik])), n_bands)
                (; ψk, εk=randn(n_bands), fk=rand(n_bands))
            end
        end
        data_mpi = random_data(basis_mpi)
        data_ref = random_data(basis_ref)

        # The full grid, and 2 k-points, which are duplicated on more than 2 ranks
        for n_kpt in (prod(kgrid), 2)
            target_mpi = explicit_full_basis(basis_mpi, kgrid; n_kpt,
                                             comm_kpts=MPI.COMM_WORLD)
            target_ref = explicit_full_basis(basis_ref, kgrid; n_kpt,
                                             comm_kpts=MPI.COMM_SELF)
            for (field, is_ψ, columns, as_view) in ((:ψk, true, :, false),
                                                    (:ψk, true, 2:3, false),
                                                    (:ψk, true, 2:3, true),
                                                    (:εk, false, :, false),
                                                    (:fk, false, :, false))
                function select(d)
                    is_ψ || return d[field]
                    as_view ? view(d[field], :, columns) : d[field][:, columns]
                end
                unfolded_mpi = DFTK.unfold_array(basis_mpi, target_mpi, select.(data_mpi), is_ψ)
                unfolded_ref = DFTK.unfold_array(basis_ref, target_ref, select.(data_ref), is_ψ)
                @test all(1:length(target_mpi.kpoints)) do ik
                    unfolded_mpi[ik] == unfolded_ref[global_index(target_mpi, ik)]
                end
                # ε and f are copied from the source; on COMM_SELF local and global indices
                # agree
                if !is_ψ
                    kmap = DFTK.unfold_mapping(basis_ref, target_ref)
                    n_kpt_ref = length(target_ref.kcoords_global)
                    @test all(1:length(target_ref.kpoints)) do it
                        σ = target_ref.kpoints[it].spin
                        ik_irred, _ = kmap[it - (σ - 1) * n_kpt_ref]
                        is = (σ - 1) * basis_ref.n_irreducible_kpoints + ik_irred
                        unfolded_ref[it] == select(data_ref[is])
                    end
                end
            end
        end
    end
end

@testitem "unfold_mapping" setup=[TestCases, UnfoldBZ] begin
    using DFTK
    using MPI
    using .UnfoldBZ: silicon_model, explicit_full_basis
    silicon = TestCases.silicon

    Ecut  = 5
    model = silicon_model(silicon, :none)
    for kgrid in ([2, 2, 2], [3, 3, 3])
        basis_mpi  = PlaneWaveBasis(model; Ecut, kgrid)
        target_mpi = explicit_full_basis(basis_mpi, kgrid; comm_kpts=MPI.COMM_WORLD)

        kmap = DFTK.unfold_mapping(basis_mpi, target_mpi)
        @test length(kmap) == length(target_mpi.kcoords_global)
        @test all(zip(kmap, target_mpi.kcoords_global)) do ((ik_irred, isym), k)
            ik_irred ≤ basis_mpi.n_irreducible_kpoints || return false
            Sk = basis_mpi.symmetries[isym].S * basis_mpi.kcoords_global[ik_irred]
            all(Δ -> abs(Δ - round(Δ)) < 1e-10, Sk - k)
        end
    end

    # Duplicated target k-points (on more than 2 ranks) map like their original
    basis = PlaneWaveBasis(model; Ecut, kgrid=[2, 2, 2])
    target = explicit_full_basis(basis, [2, 2, 2]; comm_kpts=MPI.COMM_WORLD, n_kpt=2)
    kmap = DFTK.unfold_mapping(basis, target)
    @test all(enumerate(target.kcoords_global)) do (it, k)
        kmap[it] == kmap[findfirst(==(k), target.kcoords_global)]
    end

    # The method on a single Kpoint returns the local index and the symmetry operation
    basis_ref  = PlaneWaveBasis(model; Ecut, kgrid=[3, 3, 3], comm_kpts=MPI.COMM_SELF)
    target_ref = explicit_full_basis(basis_ref, [3, 3, 3]; comm_kpts=MPI.COMM_SELF)
    @test all(target_ref.kpoints) do kpt
        ik_irred, symop = DFTK.unfold_mapping(basis_ref, kpt)
        Sk = symop.S * basis_ref.kpoints[ik_irred].coordinate
        all(Δ -> abs(Δ - round(Δ)) < 1e-10, Sk - kpt.coordinate)
    end

    basis = PlaneWaveBasis(model; Ecut, kgrid=[3, 3, 3])
    model_nosym = silicon_model(silicon, :none; symmetries=false)
    basis_outside = PlaneWaveBasis(model_nosym; Ecut, basis.fft_size,
                                   kgrid=ExplicitKpoints([[0.1, 0.2, 0.3]]))
    @test_throws ErrorException DFTK.unfold_mapping(basis, basis_outside)
end
