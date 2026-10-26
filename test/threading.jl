@testitem "DFTK threads default to Threads.nthreads() at load" tags=[:dont_test_mpi] begin
    # The package image is precompiled with one thread; the count must come from the
    # process that loads DFTK.
    code = "using DFTK; print(DFTK.get_DFTK_threads())"
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=2
           --project=$(Base.active_project()) -e $code`
    @test readchomp(cmd) == "2"
end
