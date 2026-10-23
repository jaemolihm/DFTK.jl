@testitem "Comparison of diagonalisaton procedures" tags=[:dont_test_mpi] begin
    using DFTK

    function test_solver(reference, eigensolver, prec_type)
        @testset "$eigensolver with $prec_type" begin
            nev = length(reference.λ[1])
            res = diagonalize_all_kblocks(eigensolver, reference.ham, nev; prec_type)
            @test res.λ ≈ reference.λ
        end
    end

    lattice = Float64[5 0 0; 0 0 0; 0 0 0]
    model = Model(lattice; terms=[Kinetic()])
    basis = PlaneWaveBasis(model; Ecut=100, kgrid=(1, 1, 1))
    ham = Hamiltonian(basis)
    reference = merge(diagonalize_all_kblocks(diag_full, ham, 6), (; ham))

    test_solver(reference, diag_full,    PreconditionerTPA)
    test_solver(reference, diag_full,    PreconditionerNone)
    test_solver(reference, lobpcg_hyper, PreconditionerTPA)
end

@testitem "Diagonalization with an oversized guess" setup=[TestCases] begin
    using DFTK
    using PseudoPotentialData
    silicon = TestCases.silicon

    Si = ElementPsp(silicon.atnum, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
    model = model_DFT(silicon.lattice, [Si, Si], silicon.positions;
                      functionals=[:lda_xc_teter93])
    basis = PlaneWaveBasis(model; Ecut=5, silicon.kgrid)
    ham = Hamiltonian(basis; ρ=guess_density(basis))

    nev = 6
    ref = diagonalize_all_kblocks(lobpcg_hyper, ham, 2nev; tol=1e-10)
    # Exact eigenvectors with the lowest nev in the last columns: keeping the first nev
    # columns would converge to bands nev+1:2nev, iterating the whole guess finds the
    # lowest nev.
    ψguess = [hcat(X[:, nev+1:2nev], X[:, 1:nev]) for X in ref.X]
    res = diagonalize_all_kblocks(lobpcg_hyper, ham, nev; tol=1e-8, ψguess)
    @test res.converged
    @test all(==(1), res.n_iter)
    for ik = 1:length(basis.kpoints)
        @test size(res.X[ik], 2) == nev
        @test res.λ[ik] ≈ ref.λ[ik][1:nev] atol=1e-8
    end
end
