using Knet, CUDArt, ProgressMeter
using Knet: Net, params, axpy!
if !isdefined(:MNIST)
    include(Pkg.dir("Knet/examples/mnist.jl"))
    mnist = minibatch(MNIST.xtrn, MNIST.ytrn, 100)
end

@knet function model02(x; winit=Gaussian(0,.1), hidden=64, f1=:relu, o...)
    h    = wf(x; out=hidden, f=f1, winit=winit)
    return wf(h; out=10, f=:soft, winit=winit)
end

function initmodel{T}(w::CudaArray{T}, g=similar(w); X=784, Y=10, H=div(length(w),X+Y), lr=0.5)
    f = compile(:model02; hidden=H)
    setp(f, lr=lr)
    forw(f, zeros(T, X, 1))
    back(f, zeros(T, Y, 1), softloss)
    f.reg[2].out0 = CudaArray(w.ptr, (H,X), w.dev)
    f.reg[2].dif0 = CudaArray(g.ptr, (H,X), g.dev)
    f.reg[5].out0 = CudaArray(w.ptr+H*X*sizeof(T), (Y,H), w.dev)
    f.reg[5].dif0 = CudaArray(g.ptr+H*X*sizeof(T), (Y,H), g.dev)
    return f
end

function train(f, data, loss; epochs=100)
    for epoch=1:epochs
        for (x,ygold) in mnist
            ypred = forw(f, x)
            back(f, ygold, loss)
            Knet.update!(f)
        end
        println((epoch,
                 test(f, data, softloss),
                 test(f, data, zeroone)))
    end
end

function test(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in mnist
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    return sumloss/numloss
end

interpolate(alpha, w0, w1)=axpy!(1-alpha, w0, scale!(alpha, copy(w1)))

wtest(w)=test(initmodel(w), MNIST.xtrn, softloss)

function wgrad(w)
    g = similar(w)
    f = initmodel(w, g)
    forw(f, MNIST.xtrn)
    back(f, MNIST.ytrn, softloss)
    return g
end

function plotdata(w0, w1; a=-1, b=2, s=0.1, file="plot.dat")
    xvec = Float32[]
    yvec = Float32[]
    @showprogress 1 "Plotting..." for x=a:s:b
        y = wtest(interpolate(x,w0,w1))
        push!(xvec,x)
        push!(yvec,y)
    end
    writedlm(file, [xvec yvec])
end

function doit()
    global w0=scale!(0.1, randn!(CudaArray(Float32, 794*64)))
    global w1=copy(w0)
    train(initmodel(w1), mnist, softloss)
    global s = axpy!(-1, w0, copy(w1))
    global g = wgrad(w0); scale!(-vecnorm(s)/vecnorm(g), g)
    global z = copy(w0); scale!(-vecnorm(s)/vecnorm(z), z)
    global r1 = randn!(copy(w0)); scale!(vecnorm(s)/vecnorm(r1), r1)
    global r2 = randn!(copy(w0)); scale!(vecnorm(s)/vecnorm(r2), r2)
    global r3 = randn!(copy(w0)); scale!(vecnorm(s)/vecnorm(r3), r3)
end

function doplot()
    plotdata(w0, axpy!(1,s,copy(w0)); a=0, b=1, s=0.02, file="w0w1.dat")
    plotdata(w0, axpy!(1,g,copy(w0)); a=0, b=1, s=0.02, file="w0g0.dat")
    plotdata(w0, axpy!(1,z,copy(w0)); a=0, b=1, s=0.02, file="w0z0.dat")
    plotdata(w0, axpy!(1,r1,copy(w0)); a=0, b=1, s=0.02, file="w0r1.dat")
    plotdata(w0, axpy!(1,r2,copy(w0)); a=0, b=1, s=0.02, file="w0r2.dat")
    plotdata(w0, axpy!(1,r3,copy(w0)); a=0, b=1, s=0.02, file="w0r3.dat")
    plotdata(w1, axpy!(1,r1,copy(w1)); a=0, b=1, s=0.02, file="w1r1.dat")
    plotdata(w1, axpy!(1,r2,copy(w1)); a=0, b=1, s=0.02, file="w1r2.dat")
    plotdata(w1, axpy!(1,r3,copy(w1)); a=0, b=1, s=0.02, file="w1r3.dat")
    plotdata(w1, axpy!(-1,s,copy(w1)); a=0, b=1, s=0.02, file="w1w0.dat")
    plotdata(w1, axpy!(-vecnorm(s)/vecnorm(w1),w1,copy(w1)); a=0, b=1, s=0.02, file="w1z0.dat")
end

function plot2(w0, u, d=vecnorm(u); s=d/30, noise=0, file="plot2.dat")
    u = copy(u)
    if noise > 0
        r = randn!(similar(u))
        axpy!(noise*vecnorm(u)/vecnorm(r), r, u)
    end
    u = scale!(1/vecnorm(u), u)
    xvec = Float32[]
    yvec = Float32[]
    w1 = similar(w0)
    @showprogress 1 "Plotting..." for x=0:s:d
        y = wtest(axpy!(x, u, copy!(w1,w0)))
        push!(xvec,x)
        push!(yvec,y)
    end
    writedlm(file, [xvec yvec])
    return u
end

cosine(x,y)=dot(to_host(x),to_host(y))/(vecnorm(x)*vecnorm(y))
