# Thin DFTK-flavored wrapper around the LOBPCG solver from the LOBPCGEigensolver.jl package:
# picks a sensible default tolerance and reports convergence / iteration count.
# Note that this function will return λ on the CPU,
# but X and the history on the device (for GPU runs)
@timing function lobpcg_hyper(A, X0; maxiter=100, prec=nothing,
                              tol=20size(A, 2)*eps(real(eltype(A))),
                              largest=false, n_conv_check=nothing, kwargs...)
    prec === nothing && (prec = I)

    @assert !largest "Only seeking the smallest eigenpairs is implemented."
    result = lobpcg(A, X0, I, prec, tol, maxiter; n_conv_check, timer, kwargs...)

    n_conv_check === nothing && (n_conv_check = size(X0, 2))
    n_iter = size(result.residual_history, 2) - 1
    # LOBPCG returns before maxiter only once n_conv_check vectors are locked (residual
    # < tol). Checking only the first n_conv_check of the λ-sorted output can fail when
    # n_conv_check cuts a degenerate cluster: an active member of it may be sorted in front
    # of a locked one.
    converged = n_iter < maxiter || maximum(result.residual_norms[1:n_conv_check]) < tol

    merge(result, (; n_iter, converged))
end
