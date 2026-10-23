@testitem "atomic_random_initial_guess" setup=[TestCases] begin
    using DFTK
    using PseudoPotentialData
    silicon = TestCases.silicon

    Si = ElementPsp(silicon.atnum, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    model = model_DFT(silicon.lattice, [Si, Si], silicon.positions;
                      functionals=[:lda_xc_teter93])
    basis = PlaneWaveBasis(model; Ecut=10, silicon.kgrid)

    (; orbitals) = DFTK._atomic_orbitals_nonorthogonal(basis)
    n_orbitals = 2 * DFTK.count_n_pswfc(Si.psp)
    ψ = DFTK.atomic_random_initial_guess(basis)
    @test length(ψ) == length(basis.kpoints)
    for (ik, kpt) in enumerate(basis.kpoints)
        @test size(ψ[ik]) == (length(G_vectors(basis, kpt)), n_orbitals)
        # Each coefficient is scaled by 1 + a r e^{2πiθ} with 0 ≤ r < 1
        @test all(abs.(ψ[ik] .- orbitals[ik]) .≤ 0.05 .* abs.(orbitals[ik]) .+ 1e-14)
        @test ψ[ik] != orbitals[ik]
    end
    @test DFTK.atomic_random_initial_guess(basis; amplitude=0) == orbitals

    Si_hgh = ElementPsp(silicon.atnum, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
    model_hgh = model_DFT(silicon.lattice, [Si_hgh, Si_hgh], silicon.positions;
                          functionals=[:lda_xc_teter93])
    basis_hgh = PlaneWaveBasis(model_hgh; Ecut=10, silicon.kgrid)
    @test_throws ErrorException DFTK.atomic_random_initial_guess(basis_hgh)
end

@testitem "SCF from atomic_random_initial_guess" setup=[TestCases] begin
    using DFTK
    using PseudoPotentialData
    silicon = TestCases.silicon

    Si = ElementPsp(silicon.atnum, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    model = model_DFT(silicon.lattice, [Si, Si], silicon.positions;
                      functionals=[:lda_xc_teter93], temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut=10, silicon.kgrid)
    tol = 1e-8

    ψ = DFTK.atomic_random_initial_guess(basis)
    n_bands_guess = size(ψ[1], 2)
    scfres_ref = self_consistent_field(basis; tol)

    nbandsalg = FixedBands(; n_bands_converge=5, n_bands_compute=6)
    @test nbandsalg.n_bands_compute < n_bands_guess
    scfres = self_consistent_field(basis; tol, ψ, nbandsalg)
    @test all(ψk -> size(ψk, 2) == nbandsalg.n_bands_compute, scfres.ψ)
    @test scfres.energies.total ≈ scfres_ref.energies.total atol=10tol

    nbandsalg = AdaptiveBands(model; n_bands_converge=4)
    @test nbandsalg.n_bands_compute < n_bands_guess
    scfres = self_consistent_field(basis; tol, ψ, nbandsalg)
    @test all(ψk -> size(ψk, 2) ≥ nbandsalg.n_bands_compute, scfres.ψ)
    @test scfres.energies.total ≈ scfres_ref.energies.total atol=10tol
end
