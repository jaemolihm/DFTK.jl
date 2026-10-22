@testitem "Type-stability of radial_hydrogenic" begin
    using DFTK: radial_hydrogenic

    for n = [1, 2, 3]
        @inferred radial_hydrogenic([1.0, 1.0], n)
        @inferred radial_hydrogenic([1f0, 1f0], n)
    end
end
@testitem "Rotated HydrogenicWannierProjection" setup=[TestCases] begin
    using DFTK
    using DFTK: Mat3
    using LinearAlgebra
    using Random
    silicon = TestCases.silicon

    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions; functionals=LDA())
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[1, 1, 1])
    ps = vec(map(G -> G .+ [0.1, 0.2, 0.3], G_vectors(basis)))
    center = [0.1, 0.2, 0.3]
    g(l, m; kwargs...) = DFTK.HydrogenicWannierProjection(center, 3, l, m, 1.0; kwargs...)(basis, ps)

    @testset "Identity rotation" begin
        for l = 0:3, m = -l:l
            @test g(l, m) == g(l, m; rotation=Mat3([1.0 0 0; 0 1 0; 0 0 1]))
        end
    end

    @testset "wannier90 frame x=(1,1,0), z=(0,0,1)" begin
        # R = [x̂ᵀ; ŷᵀ; ẑᵀ] with ŷ = ẑ × x̂ = (-1,1,0)/√2, so R q = ((qx+qy)/√2, (qy-qx)/√2, qz)
        x̂ = [1.0, 1.0, 0.0] / √2
        ẑ = [0.0, 0.0, 1.0]
        rotation = Mat3(vcat(x̂', cross(ẑ, x̂)', ẑ'))
        # Real harmonics m = -2:2 are dxy, dyz, dz2, dxz, dx²-y²
        dxy, dyz, dz2, dxz, dx2y2 = (g(2, m) for m = -2:2)
        @test g(2, -2; rotation) ≈ -dx2y2
        @test g(2, -1; rotation) ≈ (dyz - dxz) / √2
        @test g(2,  0; rotation) ≈ dz2
        @test g(2,  1; rotation) ≈ (dxz + dyz) / √2
        @test g(2,  2; rotation) ≈ dxy
    end

    @testset "Random rotation" begin
        rotation = Mat3(Matrix(qr(randn(Xoshiro(1), 3, 3)).Q))
        for l = 1:3
            gs  = reduce(hcat, [g(l, m) for m = -l:l])
            gsR = reduce(hcat, [g(l, m; rotation) for m = -l:l])
            # Σ_m |g_lm(q)|² is rotation invariant (addition theorem)
            @test sum(abs2, gsR; dims=2) ≈ sum(abs2, gs; dims=2)
            # g^R_lm = Σ_m' D^l_mm'(R) g_lm' (wigner_d_matrix is ill-conditioned for l = 3)
            l ≤ 2 && @test gsR ≈ gs * transpose(DFTK.wigner_d_matrix(l, rotation))
        end
    end

    @test_throws AssertionError DFTK.HydrogenicWannierProjection(center, 3, 2, 0, 1.0;
                                                                 rotation=2*one(Mat3{Float64}))
end
