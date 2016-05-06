using Knet, Optim, CUDArt, Base.LinAlg

# v is our estimated unit gradient vector, last successful step.
# "step" is a step-size/scale that we adjust.  The mean step is going to be step*v.
# we grow and shrink s using the factor "grow".
# r is a random unit vector for exploration.
# The scale for r should be proportional to the scale for v.
# Call that fixed proportionality constant rstep, i.e. step*(v+rstep*r).
# We also have rmin, i.e. when s shrinks too much, we want the exploration vector to keep a minimum size.
# step*v + max(step*rstep,rmin)*r

# We assume fmin=0 here and stop when f(x) < ftol

function dhc2{T}(f::Function, x::CudaVector{T}; grow=2.0, rmin=1e-3, rstep=1.0, ftol=1e-4)
    v = randn!(similar(x))
    r = randn!(similar(x))
    x0 = copy(x)
    f0 = f(x)
    nf = np = nfail = 1
    avgstep = 0
    while f0 > ftol
        axpy!(1, v, x)
        # we want vecnorm(r)/vecnorm(v) = rstep
        # vecnorm(randn!(r)) = sqrt(length(r))
        # scale!(rstep*vecnorm(v)/sqrt(length(r)), randn!(r))
        rscale = max(rmin, rstep*vecnorm(v))/sqrt(length(r))
        axpy!(rscale, randn!(r), x)
        f1 = f(x); nf += 1
        if f1 < f0
            nfail = 0
            copy!(x0, x)
            f0 = f1
            axpy!(rscale, r, v)
            avgstep = (avgstep==0 ? vecnorm(v) : 0.99*avgstep + 0.01*vecnorm(v))
            scale!(grow, v)
        else
            nfail += 1
            copy!(x, x0)
            scale!(1/grow, v)
        end
        nf >= np && (println((nf,f0,avgstep,x[1],test02(mnist_f, mnist_tst, zeroone) )); np *= 2)
    end
    (nf,f0,avgstep,x)
end

# Set up mnist as an objective function

using Knet
using Knet: Net, params, axpy!
if !isdefined(:MNIST)
    include(Pkg.dir("Knet/examples/mnist.jl"))
end
batchsize = 1000
mnist_trn = minibatch(MNIST.xtrn, MNIST.ytrn, 60000)
mnist_tst = minibatch(MNIST.xtst, MNIST.ytst, 10000)

@knet function model02(x; winit=Gaussian(0,.1), hidden=32, f1=:relu, o...)
    h    = wf(x; out=hidden, f=f1, winit=winit)
    return wf(h; out=10, f=:soft, winit=winit)
end

# We'll try to create the two weight matrices from a single array

function initmodel{T}(w::CudaArray{T}; X=784, H=32, Y=10, B=100)
    @assert length(w) == X*H+H*Y
    f = compile(:model02; hidden=H)
    forw(f, zeros(T, X, B))
    f.reg[2].out0 = CudaArray(w.ptr, (H,X), w.dev)
    f.reg[5].out0 = CudaArray(w.ptr+H*X*sizeof(T), (Y,H), w.dev)
    return f
end

mnist_w = fill!(CudaArray(Float32,794*32),0)
mnist_f = initmodel(mnist_w)

function mnist_loss1(w)
    global mnist_f, mnist_w
    w === mnist_w || error("w")
    (x,y) = mnist_trn[1]
    softloss(forw(mnist_f, x), y)
end

function test02(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    sumloss / numloss
end

# dhc2(mnist_loss1, mnist_w)
# test02(mnist_f, mnist_tst, softloss)
