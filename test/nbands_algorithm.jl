@testitem "SCF band count does not follow the column count of ψ" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                      functionals=LDA(), temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut=10, silicon.kgrid)
    tol = 1e-8
    n_bands_guess = 12
    ψ = DFTK.random_orbitals(basis, n_bands_guess)

    nbandsalg = FixedBands(; n_bands_converge=5, n_bands_compute=6)
    scfres = self_consistent_field(basis; tol, ψ, nbandsalg)
    @test scfres.converged
    @test all(ψk -> size(ψk, 2) == nbandsalg.n_bands_compute, scfres.ψ)

    # The first SCF step computes n_bands_compute bands, not one per column of ψ
    nbandsalg = AdaptiveBands(model; n_bands_converge=4)
    @test nbandsalg.n_bands_compute < n_bands_guess
    n_bands_first = Ref(0)
    callback = info -> (iszero(n_bands_first[]) && (n_bands_first[] = size(info.ψ[1], 2)))
    scfres = self_consistent_field(basis; tol, ψ, nbandsalg, callback)
    @test scfres.converged
    @test n_bands_first[] == nbandsalg.n_bands_compute
end
