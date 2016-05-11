# implement dhc of degree n where n=0 means random search, n=1 means
# use one previous direction etc.

using Knet
@useifgpu CUDArt
# using ProgressMeter: @showprogress, Progress, next!
using Knet: Net, params, axpy!, copysync!, BaseArray
typealias BaseVector{T} BaseArray{T,1}

if !isdefined(:MNIST)
    include(Pkg.dir("Knet/examples/mnist.jl"))
    mnist100 = minibatch(MNIST.xtrn, MNIST.ytrn, 100)
end

@knet function model02(x; winit=Gaussian(0,.1), hidden=64, f1=:relu, o...)
    h    = wf(x; out=hidden, f=f1, winit=winit)
    return wf(h; out=10, f=:soft, winit=winit)
end

cosine(x,y)=CUBLAS.dot(x,y)/(vecnorm(x)*vecnorm(y))
setseed(42)
fmodel = nothing
gc()
@gpu w64 = scale!(0.01, randn!(CudaArray(Float32, 794*64)))
@gpu w32 = scale!(0.01, randn!(CudaArray(Float32, 794*32)))
w64cpu = scale!(0.01, randn!(Array(Float32, 794*64)))
w32cpu = scale!(0.01, randn!(Array(Float32, 794*32)))


@gpu function initmodel{T}(w::CudaArray{T}, g=nothing; X=784, Y=10, H=div(length(w),X+Y), lr=0.5)
    global fmodel
    if fmodel == nothing
        fmodel = compile(:model02; hidden=H)
        setp(fmodel, lr=lr)
        forw(fmodel, zeros(T, X, 1))
        back(fmodel, zeros(T, Y, 1), softloss)
    end
    fmodel.reg[2].out = fmodel.reg[2].out0 = CudaArray(w.ptr, (H,X), w.dev)
    fmodel.reg[5].out = fmodel.reg[5].out0 = CudaArray(w.ptr+H*X*sizeof(T), (Y,H), w.dev)
    if g != nothing
        fmodel.reg[2].dif = fmodel.reg[2].dif0 = CudaArray(g.ptr, (H,X), g.dev)
        fmodel.reg[5].dif = fmodel.reg[5].dif0 = CudaArray(g.ptr+H*X*sizeof(T), (Y,H), g.dev)
    end
    return fmodel
end

submatrix(a, i, m, n)=pointer_to_array(pointer(a, i), (m, n))

function initmodel{T}(w::Array{T}, g=nothing; X=784, Y=10, W=length(w), H=div(W,X+Y), lr=0.5)
    global fmodel
    if fmodel == nothing
        fmodel = compile(:model02; hidden=H)
        setp(fmodel, lr=lr)
        forw(fmodel, zeros(T, X, 1))
        back(fmodel, zeros(T, Y, 1), softloss)
    end
    if pointer(w) != pointer(fmodel.reg[2].out0)
        fmodel.reg[2].out = fmodel.reg[2].out0 = submatrix(w, 1, H, X) # CudaArray(w.ptr, (H,X), w.dev)
        fmodel.reg[5].out = fmodel.reg[5].out0 = submatrix(w, 1+H*X, Y, H) # CudaArray(w.ptr+H*X*sizeof(T), (Y,H), w.dev)
    end
    if g != nothing && pointer(g) != pointer(fmodel.reg[2].dif0)
        fmodel.reg[2].dif = fmodel.reg[2].dif0 = submatrix(g, 1, H, X) # CudaArray(g.ptr, (H,X), g.dev)
        fmodel.reg[5].dif = fmodel.reg[5].dif0 = submatrix(g, 1+H*X, Y, H) # CudaArray(g.ptr+H*X*sizeof(T), (Y,H), g.dev)
    end
    return fmodel
end

# function wtest2(w, loss=softloss)
#     f = initmodel(w)
#     y = forw(f, MNIST.xtst)
#     return loss(y, MNIST.ytst)
# end

# function wgrad(w, g=similar(w); x=MNIST.xtrn, y=MNIST.ytrn, loss=softloss)
#     f = initmodel(w, g)
#     ypred = forw(f, x)
#     back(f, y, loss)
#     return loss(ypred, y)
# end

function weval(w, g=nothing; x=MNIST.xtrn, y=MNIST.ytrn, loss=softloss)
    f = initmodel(w, g)
    ypred = forw(f, x)
    g != nothing && back(f, y, loss)
    return loss(ypred, y)
end

function wtest(w, loss=softloss; x=MNIST.xtrn, y=MNIST.ytrn)
    f = initmodel(w)
    ypred = forw(f, x)
    return loss(ypred, y)
end

function gradest0{T}(f::Function, x::BaseVector{T}; rscale=0.005, nkeep=5)
    g0,g1,x0
    x0 = copy(x)
    g0 = similar(x)
    f0 = f(x0, g0)
    xdims = length(x)
    xkeep = zeros(T, xdims, nkeep)
    fkeep = zeros(T, 1, nkeep)
    r = similar(x)
    for i=1:nkeep
        f1 = f(axpy!(rscale, randn!(r), copy!(x,x0)))
        fkeep[i] = f1
        copysync!(xkeep, 1+(i-1)*xdims, x, 1, xdims)
    end
    xdiff = broadcast(-, xkeep, to_host(x0)) # X,N
    fdiff = fkeep - f0                       # 1,N
    g1 = vec(fdiff / xdiff)
    g0 = to_host(g0)
    n0 = vecnorm(g0)
    n1 = vecnorm(g1)
    cosine = dot(g0,g1)/(n0*n1)
    println((:nkeep,nkeep,:rscale,rscale,:cos,cosine,:n0,n0,:n1,n1))
end

function gradest1{T}(f::Function, x::BaseVector{T}; rscale=0.005, nkeep=5)
    g0,g1,x0
    x0 = copy(x)
    g0 = similar(x)
    f0 = f(x0, g0)
    xdims = length(x)
    xkeep = zeros(T, xdims, nkeep)
    fkeep = zeros(T, 1, nkeep)
    r = similar(x)
    for i=1:nkeep
        f1 = f(axpy!(rscale, randn!(r), copy!(x,x0)))
        fkeep[i] = f1
        copysync!(xkeep, 1+(i-1)*xdims, x, 1, xdims)
    end
    g1 = vec(fkeep / xkeep)
    g0 = to_host(g0)
    n0 = vecnorm(g0)
    n1 = vecnorm(g1)
    cosine = dot(g0,g1)/(n0*n1)
    println((:nkeep,nkeep,:rscale,rscale,:cos,cosine,:n0,n0,:n1,n1))
end

function gradest2{T}(f::Function, x::BaseVector{T}; rscale=0.05, nkeep=5)
    g0,g1,x0
    x0 = copy(x)
    g0 = similar(x)
    f0 = f(x0, g0)
    xdims = length(x)
    xkeep = ones(T, 1+xdims, nkeep)
    fkeep = zeros(T, 1, nkeep)
    r = similar(x)
    for i=1:nkeep
        f1 = f(axpy!(rscale, randn!(r), copy!(x,x0)))
        fkeep[i] = f1
        copysync!(xkeep, 2+(i-1)*(1+xdims), x, 1, xdims)
    end
    g1 = vec(fkeep / xkeep)[2:end]
    g0 = to_host(g0)
    n0 = vecnorm(g0)
    n1 = vecnorm(g1)
    cosine = dot(g0,g1)/(n0*n1)
    println((:nkeep,nkeep,:rscale,rscale,:cos,cosine,:n0,n0,:n1,n1))
end

function gradest3{T}(f::Function, x::BaseVector{T}; rscale=0.005, nkeep=5)
    g0,g1,x0,xdiff,fdiff
    x0 = copy(x)
    g0 = similar(x)
    f0 = f(x0, g0)
    xdims = length(x)
    xkeep = zeros(T, xdims, nkeep+1)
    fkeep = zeros(T, 1, nkeep+1)
    copysync!(xkeep, 1, x0, 1, xdims)
    fkeep[1] = f0
    npair = div(nkeep*(nkeep+1),2)
    xdiff = zeros(T, xdims, npair)
    fdiff = zeros(T, 1, npair)
    idiff = 0
    r = similar(x)
    for i=2:nkeep+1
        f1 = f(axpy!(rscale, randn!(r), copy!(x,x0)))
        fkeep[i] = f1
        copysync!(xkeep, 1+(i-1)*xdims, x, 1, xdims)
        for j=1:i-1
            dx1 = xkeep[:,i] - xkeep[:,j]
            copysync!(xdiff, 1+idiff*xdims, dx1, 1, xdims)
            fdiff[idiff+=1] = f1 - fkeep[j]
        end
    end
    g1 = vec(fdiff / xdiff)
    g0 = to_host(g0)
    n0 = vecnorm(g0)
    n1 = vecnorm(g1)
    cosine = dot(g0,g1)/(n0*n1)
    println((:nkeep,nkeep,:rscale,rscale,:cos,cosine,:n0,n0,:n1,n1))
end

function gradest4{T}(f::Function, x::BaseVector{T}; rscale=0.0001, nsample=256, lr=1.0, l2=0.0)
    g0,x0,f0,n0,g,r,f1,f2,cs,n1,nr
    x0 = copy(x)
    g0 = similar(x)
    f0 = f(x0, g0)
    n0 = vecnorm(g0)
    nx = vecnorm(x0)
    @show CUBLAS.dot(x0,g0)/(n0*nx)
    r = similar(x)
    g = fill!(similar(x),0)
    nf = 0; np = 1
    for i=1:nsample
        scale!(rscale, randn!(r))
        nr = vecnorm(r)
        f1 = f(axpy!(1, r, copy!(x,x0)))
        f2 = f0 + CUBLAS.dot(g, r)
        scale!(1-lr*l2, g)
        axpy!(-lr*(f2-f1)/(nr*nr), r, g) # NLMS algorithm normalizes with nr*nr (wikipedia, which also has convergence proofs)
        n1 = vecnorm(g)
        cs = CUBLAS.dot(g0,g)/(n0*n1)
        if (nf+=1) >= np
            println((i,:cos,cs,:n1n0,n1/n0,:f2f1,f2-f1,:f1f0,f1-f0,:f0,f0,:n0,n0,:n1,n1,:nr,nr))
            np *= 2
        end
    end
    println((nsample,:cos,cs,:n1n0,n1/n0,:f2f1,f2-f1,:f1f0,f1-f0,:f0,f0,:n0,n0,:n1,n1,:nr,nr))
end

function train(f, data, loss; epochs=100)
    p = Progress(epochs, dt=1, barlen=40)
    for epoch=1:epochs
        for (x,ygold) in data
            ypred = forw(f, x)
            back(f, ygold, loss)
            update!(f)
        end
        zloss = test(f, data, zeroone)
        sloss = test(f, data, softloss)
        p.desc = @sprintf("%d/%d %.6f %.6f ", epoch, epochs, sloss, zloss)
        next!(p)
    end
end

function test(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    return sumloss/numloss
end

# inc!(d,k)=(d[k]=1+get(d,k,0))
# avg!(d,k,v;a=0.01)=(d[k]=a*v+(1-a)*get(d,k,v))

function dhc0{T}(f::Function, x::BaseVector{T}; grow=2.0, rmin=1e-5, rscale=0.05, ftol=1e-6, smin=0.05, smax=0.45, salpha=0.1)
    r = similar(x)
    x0 = copy(x)
    f0 = f(x)
    savg = (smin+smax)/2
    nf = np = 1
    while f0 > ftol
        f1 = f(axpy!(rscale, randn!(r), copy!(x,x0)))
        if f1 < f0
            f0 = f1
            copy!(x0, x)
            savg = (1-salpha)*savg + salpha
        else
            savg = (1-salpha)*savg
        end
        if savg > smax
            # print("+")
            rscale *= grow
            savg = (smin+smax)/2
        elseif savg < smin
            # print("-")
            rscale /= grow
            rscale < rmin && break
            savg = (smin+smax)/2
        end
        if (nf+=1) >= np
            println((nf,:f0,f0,:rscale,rscale,:savg,savg,:x1,x0[1],:err,wtest(x0, zeroone)))
            np *= 2
        end
        if nf % 1000 == 0
            gc()
            # println((:mem, Knet.gpumem(), :gc, (gc();gpusync();Knet.gpumem())))
        end
    end
    println((nf,:f0,f0,:rscale,rscale,:savg,savg,:x1,x0[1],:err,wtest(x0, zeroone)))
    copy!(x,x0)
    return f0
end

Knet.copysync!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)=copy!(dest,doffs,src,soffs,n)
# @gpu import CUDArt: to_host
# to_host(x)=x

function try_step{T}(f::Function, x::BaseVector{T}, step::BaseVector{T}, scale::Float64, x0::BaseVector{T}, f0::Float64, xkeep::Matrix{T}, fkeep::Matrix{T}, ikeep::Int)
    xdims = length(x)
    f1 = f(axpy!(scale, step, copy!(x, x0)))
    if f1 > f0
        fkeep[ikeep] = f1
        copysync!(xkeep, 1+(ikeep-1)*xdims, x, 1, xdims)
    else
        fkeep[ikeep] = f0
        copysync!(xkeep, 1+(ikeep-1)*xdims, x0, 1, xdims)
        copysync!(x0, x)
        f0 = f1
    end
    return f0
end

function dhc1{T}(f::Function, x::BaseVector{T}; nkeep=5, grow=2.0, rmin=1e-5, rscale=0.05, ftol=1e-6, smin=0.05, smax=0.45, salpha=0.1)
    # Keep the last few points and function values:
    xdims = length(x)
    xkeep = zeros(T, xdims, nkeep)
    fkeep = zeros(T, 1, nkeep)

    # Initialize the buffer:
    r = similar(x)
    x0 = copy(x)
    f0 = f(x); nf = np = 1
    for i = 1:nkeep
        f0 = try_step(f, x, randn!(r), rscale, x0, f0, xkeep, fkeep, i); nf+=1
    end

    fdiff = similar(fkeep)
    xdiff = similar(xkeep)
    fgrad = similar(x)
    savg = 0

    while f0 > ftol
        # Estimate the gradient at x0; the / operator solves linear regression:
        broadcast!(-, xdiff, xkeep, to_host(x0))
        fdiff = fkeep - f0
        copy!(fgrad, xdiff / fdiff)

        # It turns out the ideal learning rate is fairly stable around 1.0
        # fscale = rscale * sqrt(xdims) / vecnorm(fgrad)
        fscale = 1.0

        # Try a step opposite the gradient direction
        f0 = try_step(f, x, fgrad, -fscale, x0, f0, xkeep, fkeep, rand(1:nkeep)); nf+=1
        # Try a step in a random direction
        f0 = try_step(f, x, randn!(r), rscale, x0, f0, xkeep, fkeep, rand(1:nkeep)); nf+=1

        # Report
        if (nf+=1) >= np
            println((nf,:f0,f0,:rscale,rscale,:savg,savg,:x1,x0[1],:err,wtest(x0, zeroone)))
            np *= 2
        end

        # Rescale (based on dhc0, TODO: make this adaptive)
        rscale = (nf < 32 ? 0.05 : nf < 512 ? 0.025 : nf < 2048 ? 0.0125 : 0.00625)
    end
    println((nf,:f0,f0,:rscale,rscale,:savg,savg,:x1,x0[1],:err,wtest(x0, zeroone)))
    copy!(x,x0)
    return f0
end

function gd0{T}(f::Function, x::BaseVector{T}, g::BaseVector{T}=similar(x); gscale=1.5, ftol=0.05, alpha=0.1)
    f0 = f(x, g)
    nf = np = 1
    savg = 1.0
    gavg = vecnorm(g)
    while f0 > ftol
        f1 = f(axpy!(-gscale, g, x), g)
        savg = (1-alpha)*savg + alpha * (f1<f0)
        gavg = (1-alpha)*gavg + alpha * vecnorm(g)
        f0 = f1
        if (nf+=1) >= np
            println((nf,:f0,f0,:gscale,gscale,:savg,savg,:gavg,gavg,:err,wtest(x, zeroone)))
            np *= 2
        end
        if nf % 1000 == 0
            gc()
            # println((:mem, Knet.gpumem(), :gc, (gc();gpusync();Knet.gpumem())))
        end
    end
    println((nf,:f0,f0,:gscale,gscale,:savg,savg,:gavg,gavg,:err,wtest(x, zeroone)))
    return f0
end

function gd1{T}(f::Function, fgrad::Function, x::BaseVector{T}, g::BaseVector{T}=similar(x); grow=sqrt(2.0), gmin=1e-5, gscale=1.0, ftol=0.05, lravg=gscale, galpha=0.1)
    x0 = copy(x)
    f0 = fgrad(x, g)
    nf = np = 1
    gnorm = vecnorm(g)
    while f0 > ftol
        f1 = f(axpy!(-gscale, g, copy!(x,x0)))
        while f1 < f0
            gscale *= grow
            f1 = f(axpy!(-gscale, g, copy!(x,x0)))
        end
        while f1 > f0
            gscale /= grow
            f1 = f(axpy!(-gscale, g, copy!(x,x0)))
        end
        copy!(x0, x)
        f0 = fgrad(x0, g)
        @assert f0==f1
        lravg = galpha * gscale + (1-galpha) * lravg
        gnorm = galpha * vecnorm(g) + (1-galpha) * gnorm
        if (nf+=1) >= np
            println((nf,:f0,f0,:lravg,lravg,:gnorm,gnorm,:err,wtest(x0, zeroone)))
            np *= 2
        end
        if nf % 1000 == 0
            gc() # println((:mem, Knet.gpumem(), :gc, (gc();gpusync();Knet.gpumem())))
        end
    end
    println((nf,:f0,f0,:lravg,lravg,:gnorm,gnorm,:err,wtest(x0, zeroone)))
    copy!(x,x0)
    return f0
end

# Test gradient estimation using SGD updates
# using CUBLAS: dot

function gd2{T}(f::Function, x::BaseVector{T}, g::BaseVector{T}=similar(x); gscale=1.5, ftol=0.05, alpha=0.1, lr=0.0001, l2=0)
    f0 = f(x, g)
    nf = np = 1
    savg = 1.0
    gavg = vecnorm(g)

    # sgd update: g := (1-lr*h) g - lr * (g.w-f) * w

    # f(w) = g.w  # assuming loss is linear in w
    # df/dw = g   # g is also the gradient
    # w' = w - lr*g # therefore would be used in updating w in SGD

    # Finding g with online linear regression (sgd)
    # J = (1/2) (g.w-f)^2 + (h/2) |g|^2  # regularized loss from predicting f(w)=g.w
    # dJ/dg = (g.w - f) * w + h * g
    # g' = g - lr * ((g.w - f) * w + h * g)
    # g' = (1-lr*h) g - lr * (g.w-f) * w

    ghat = fill!(similar(g),0)
    axpy!(lr * f0, x, ghat)
    n0_avg = vecnorm(g)
    n1_avg = vecnorm(ghat)
    cs_avg = CUBLAS.dot(g,ghat)/(n0_avg*n1_avg)
    
    while f0 > ftol
        f1 = f(axpy!(-gscale, g, x), g)
        savg = (1-alpha)*savg + alpha * (f1<f0)
        gavg = (1-alpha)*gavg + alpha * vecnorm(g)
        f0 = f1

        f2 = CUBLAS.dot(ghat, x)
        scale!(1-lr*l2, ghat)
        axpy!(-lr*(f2-f0), x, ghat)

        n0 = vecnorm(g)
        n1 = vecnorm(ghat)
        cs = CUBLAS.dot(g,ghat)/(n0*n1)
        n0_avg = (1-alpha)*n0_avg + alpha * n0
        n1_avg = (1-alpha)*n1_avg + alpha * n1
        cs_avg = (1-alpha)*cs_avg + alpha * cs
        
        if (nf+=1) >= np
            println((nf,:lr,lr,:l2,l2,:cs,cs_avg,:n0,n0_avg,:n1,n1_avg,:f0,f0,:gscale,gscale,:savg,savg,:gavg,gavg,:err,wtest(x, zeroone)))
            np *= 2
        end
        if nf % 1000 == 0
            gc()
            # println((:mem, Knet.gpumem(), :gc, (gc();gpusync();Knet.gpumem())))
        end
    end
    println((nf,:lr,lr,:l2,l2,:cs,cs_avg,:n0,n0_avg,:n1,n1_avg,:f0,f0,:gscale,gscale,:savg,savg,:gavg,gavg,:err,wtest(x, zeroone)))
    return f0
end


# gd2 does not handle the biases right.  It is using g.x=f.  It should
# be using g.x+b=f with b approximately equal to the value of the
# first point.  Or instead we can use (g.x1+b)-(g.x2+b)=(f1-f2) or
# g.(x1-x2)=(f1-f2) for updates and not worry about biases.  This is gd3.

function gd3{T}(f::Function, x::BaseVector{T}; gscale=1.5, ftol=0.05, alpha=0.01, lr=1.0, l2=0.1)
    g0 = similar(x)
    f0 = f(x, g0)

    nf = np = 1
    savg = 1.0
    gavg = vecnorm(g0)

    g1 = fill!(similar(g0),0)
    n0_avg = vecnorm(g0)
    n1_avg = 0
    cs_avg = 1
    
    while f0 > ftol
        f1 = f(axpy!(-gscale, g0, x), g0)
        savg = (1-alpha)*savg + alpha * (f1<f0)
        gavg = (1-alpha)*gavg + alpha * vecnorm(g0)

        # Try to estimate the gradient and report the cosine.  Note that we are moving using the real gradient.

        # dx = x - x0 = -gscale*g0
        # f1hat = f0 + g1.(x-x0)
        f1hat = f0 - gscale * CUBLAS.dot(g1, g0)
        # dg = (f1hat-f1)*(x-x0) = (f1hat-f1)*(-gscale)*g0
        # update g = g - lr * dg = g + lr * (f1hat-f1) * gscale * g0
        scale!(1-lr*l2, g1)
        axpy!(lr*(f1hat-f1)*gscale, g0, g1)

        f0 = f1
        
        n0 = vecnorm(g0)
        n1 = vecnorm(g1)
        cs = CUBLAS.dot(g0,g1)/(n0*n1)
        n0_avg = (1-alpha)*n0_avg + alpha * n0
        n1_avg = (1-alpha)*n1_avg + alpha * n1
        cs_avg = (1-alpha)*cs_avg + alpha * cs
        
        if (nf+=1) >= np
            println((nf,:lr,lr,:l2,l2,:cs,cs_avg,:n0,n0_avg,:n1,n1_avg,:f0,f0,:gscale,gscale,:savg,savg,:gavg,gavg,:err,wtest(x, zeroone)))
            np *= 2
        end
        if nf % 1000 == 0
            gc()
            # println((:mem, Knet.gpumem(), :gc, (gc();gpusync();Knet.gpumem())))
        end
    end
    println((nf,:lr,lr,:l2,l2,:cs,cs_avg,:n0,n0_avg,:n1,n1_avg,:f0,f0,:gscale,gscale,:savg,savg,:gavg,gavg,:err,wtest(x, zeroone)))
    return f0
end

# Run gd, report how much the gradient turns
function gd3turn{T}(f::Function, x::BaseVector{T}; gscale=1.5, ftol=0.05, alpha=0.1)
    g0 = similar(x)
    g1 = similar(x)
    f0 = f(x, g0)
    n0 = gavg = vecnorm(g0)
    cavg = savg = NaN
    nf = np = 1
    while f0 > ftol
        f1 = f(axpy!(-gscale, g0, x), g1)
        n1 = vecnorm(g1)
        gavg = (1-alpha)*gavg + alpha * n1
        savg = isnan(savg) ? 1.0*(f1<f0) : (1-alpha)*savg + alpha * (f1<f0)
        g0g1 = CUBLAS.dot(g0,g1)/(n0*n1)
        cavg = isnan(cavg) ? g0g1 : (1-alpha)*cavg + alpha * g0g1
        (nf+=1) >= np && (np*=2; println((nf,:f0,f0,:cavg,cavg,:savg,savg,:gavg,gavg,:err,wtest(x, zeroone))))
        nf % 100 == 0 && gc()
        copy!(g0,g1); n0,f0 = n1,f1
    end
    println((nf,:f0,f0,:cavg,cavg,:savg,savg,:gavg,gavg,:err,wtest(x, zeroone)))
    return f0
end

function ego2{T}(f::Function, x::BaseVector{T}; gscale=1.0, ftol=0.05, alpha=0.01, lr=1.0, l2=0.1, noise=1.0, rmin=1e-6, stepsize=100.0)
    # x0 is the anchor point
    x0 = copy(x)
    g0 = similar(x0)
    f0 = f(x0, g0)
    n0 = vecnorm(g0)
    xdims = length(x0)

    # g1 is the gradient estimate, starts at 0
    g1 = fill!(similar(g0),0)
    n1 = 0

    # arrays for actual gradient (for debugging) and step
    g = similar(x)
    r = similar(x)

    # some statistics
    nf = np = 1
    savg = 0.5
    n0_avg = n0
    n1_avg = n1
    nr_avg = rmin*sqrt(xdims)
    cs_avg = 0.0
    
    while f0 > ftol
        # rmin is the minimum scale of the random vector components
        # noise is the norm ratio of random vector to gradient vector
        rscale = max(rmin, noise*n1/sqrt(xdims))
        scale!(rscale, randn!(r))

        # add noise to the negative gradient to get the step vector
        axpy!(-1, g1, r)
        # take step scaled with stepsize (default 1), updating f1 and g
        scale!(stepsize, r)
        nr = vecnorm(r)
        isfinite(nr)  || error("!isfinite(r)")
        f1 = f(axpy!(1, r, copy!(x,x0)), g)
        # compute success rate
        savg = (1-alpha)*savg + alpha * (f1<f0)

        # dx = x - x0 = r
        # f2 = f0 + g1.(x-x0) is our estimate
        f2 = f0 + CUBLAS.dot(g1, r)
        # weight decay:
        l2!=0 && scale!(1-lr*l2, g1)
        # dg = (f2-f1)*(x-x0) = (f2-f1)*r
        # nlms update g1 = g1 - (lr/nr^2) * dg = g1 - (lr/nr^2) * (f2-f1) * r
        axpy!(-lr*(f2-f1)/(nr*nr), r, g1)
        n1 = vecnorm(g1)
        isfinite(n1) || error("!isfinite(n1)")

        if f1 < f0
            copy!(x0, x)
            copy!(g0, g)
            f0 = f1
            n0 = vecnorm(g0)
        end
        
        # Report
        cs = n1==0 ? 0 : CUBLAS.dot(g0,g1)/(n0*n1)
        isfinite(cs) || error("!isfinite(cs)")
        n0_avg = (1-alpha)*n0_avg + alpha * n0
        n1_avg = (1-alpha)*n1_avg + alpha * n1
        nr_avg = (1-alpha)*nr_avg + alpha * nr
        cs_avg = (1-alpha)*cs_avg + alpha * cs
        if (nf+=1) >= np
            # println((i,:cos,cs,:n1n0,n1/n0,:f2f1,f2-f1,:f1f0,f1-f0,:f0,f0,:n0,n0,:n1,n1,:nr,nr))
            println((nf,:f0,f0,:cos,cs_avg,:n0,n0_avg,:n1,n1_avg,:nr,nr_avg,:savg,savg,:err,wtest(x, zeroone),:lr,lr,:l2,l2,:stepsize,stepsize,:noise,noise))
            np *= 2
        end
        if nf % 1000 == 0
            gc()
            # println((:mem, Knet.gpumem(), :gc, (gc();gpusync();Knet.gpumem())))
        end
    end
    println((nf,:f0,f0,:cos,cs_avg,:n0,n0_avg,:n1,n1_avg,:nr,nr_avg,:savg,savg,:err,wtest(x, zeroone),:lr,lr,:l2,l2,:stepsize,stepsize,:noise,noise))
    return f0
end

# Make the step size (maybe later also the noise size) adaptive.

function ego3{T}(f::Function, x::BaseVector{T}; ftol=0.05, alpha=0.01, lr=0.5, l2=0.0, noise=2.0, rmin=1e-6, stepsize=128.0, smin=0.2, smax=0.8, salpha=0.1, grow=sqrt(2.0), maxnf=Inf)
    # x0 is the anchor point
    x0 = copy(x)
    g0 = similar(x0)
    f0 = f(x0, g0)
    n0 = vecnorm(g0)
    xdims = length(x0)

    # g1 is the gradient estimate, starts at 0
    g1 = fill!(similar(g0),0)
    n1 = 0

    # arrays for actual gradient (for debugging) and step
    # TODO: we should measure the angle to target rather than the (ever changing) local gradient.
    g = similar(x)
    r = similar(x)

    # some statistics
    nf = 1; t0 = 0
    savg = 0.5
    n0_avg = n0
    n1_avg = n1
    nr_avg = rmin*sqrt(xdims)
    cs_avg = 0.0
    
    while f0 > ftol
        # rmin is the minimum scale of the noise vector components
        # noise is the norm ratio of noise vector to gradient vector
        rscale = max(rmin, noise*n1/sqrt(xdims))
        scale!(rscale, randn!(r)) # this makes vecnorm(r) ~ noise*vecnorm(g1)
        # add noise to the negative gradient to get the step vector and scale with stepsize
        axpy!(-1, g1, r)
        scale!(stepsize, r)
        nr = vecnorm(r)

        # take step scaled with stepsize, updating f1 and g
        f1 = f(axpy!(1, r, copy!(x,x0)), g)
        nf += 1

        # compute success rate and update stepsize accordingly
        savg = (1-salpha)*savg + salpha * (f1<f0)
        savg < smin && (stepsize /= grow; savg = 0.5)
        savg > smax && (stepsize *= grow; savg = 0.5)

        # update gradient estimate
        # f2 = f0 + g1.(x-x0) is our fval estimate
        # note that r = x - x0
        f2 = f0 + CUBLAS.dot(g1, r)
        # weight decay:
        l2!=0 && scale!(1-lr*l2, g1)
        # J = (f2-f1)^2/2
        # dg = dJ/dg1 = (f2-f1)*(x-x0) = (f2-f1)*r
        # nlms update g1 = g1 - lr * dg * (1/nr^2) = g1 - lr * (f2-f1) * r * (1/nr^2)
        axpy!(-lr*(f2-f1)/(nr*nr), r, g1)
        n1 = vecnorm(g1)

        if f1 < f0
            copy!(x0, x)
            copy!(g0, g)
            f0 = f1
            n0 = vecnorm(g0)
        end
        
        # Report
        cs = (n1==0) ? 0 : CUBLAS.dot(g0,g1)/(n0*n1)
        n0_avg = (1-alpha)*n0_avg + alpha * n0
        n1_avg = (1-alpha)*n1_avg + alpha * n1
        nr_avg = (1-alpha)*nr_avg + alpha * nr
        cs_avg = (1-alpha)*cs_avg + alpha * cs
        if time() >= t0
            println((nf,:f0,f0,:cos,cs_avg,:n0,n0_avg,:n1,n1_avg,:nr,nr_avg,:savg,savg,:err,wtest(x, zeroone),:lr,lr,:l2,l2,:stepsize,stepsize,:noise,noise))
            t0 = 10+time()
        end
        nf % 100 == 0 && gc()
        nf >= maxnf && break
    end
    println((nf,:f0,f0,:cos,cs_avg,:n0,n0_avg,:n1,n1_avg,:nr,nr_avg,:savg,savg,:err,wtest(x, zeroone),:lr,lr,:l2,l2,:stepsize,stepsize,:noise,noise))
    return f0
end

urand(min,max)=rand()*(max-min)+min
lrand(min,max)=exp(urand(log(min),log(max)))
function test_ego3()
    copy!(w64, scale!(0.01, convert(Array{Float32}, randn(MersenneTwister(42), size(w64)))))
    lr=lrand(1.2/2, 1.2*2)
    l2=(rand()<0.5 ? 0 : lrand(0.001/lr,1.0/lr))
    noise=lrand(1.6/2, 1.6*2)
    stepsize=lrand(100/2, 100*2)
    @show (lr,l2,noise,stepsize)
    ego3(weval,w64; maxnf=1000, lr=lr, l2=l2, noise=noise, stepsize=stepsize)
end


# Separate gradient steps and noise steps

function ego4{T}(f::Function, x::BaseVector{T}; ftol=0.05, alpha=0.01, lr=0.5, l2=0.0, noise=2.0, rmin=1e-3, stepsize=128.0, smin=0.2, smax=0.8, salpha=0.1, grow=sqrt(2.0), maxnf=Inf, rstep=2)
    # x0 is the anchor point
    x0 = copy(x)
    g0 = similar(x0)
    f0 = f(x0, g0)
    n0 = vecnorm(g0)
    xdims = length(x0)

    # g1 is the gradient estimate, starts at 0
    g1 = fill!(similar(g0),0)
    n1 = 0

    # arrays for actual gradient (for debugging) and step
    # TODO: we should measure the angle to target rather than the (ever changing) local gradient.
    g = similar(x)
    r = similar(x)

    # some statistics
    nf = 1; t0 = 0
    savg = 0.5
    n0_avg = n0
    n1_avg = n1
    nr_avg = rmin*sqrt(xdims)
    cs_avg = 0.0
    
    while f0 > ftol
        if n1==0 || nf%rstep==0
            scale!(rmin, randn!(r))
        else
            axpy!(-stepsize, g1, r)
        end
        nr = vecnorm(r)

        # take step scaled with stepsize, updating f1 and g
        f1 = f(axpy!(1, r, copy!(x,x0)), g)
        nf += 1

        # compute success rate and update stepsize accordingly
        savg = (1-salpha)*savg + salpha * (f1<f0)
        savg < smin && (stepsize /= grow; savg = 0.5)
        savg > smax && (stepsize *= grow; savg = 0.5)

        # update gradient estimate
        # f2 = f0 + g1.(x-x0) is our fval estimate
        # note that r = x - x0
        f2 = f0 + CUBLAS.dot(g1, r)
        # weight decay:
        l2!=0 && scale!(1-lr*l2, g1)
        # J = (f2-f1)^2/2
        # dg = dJ/dg1 = (f2-f1)*(x-x0) = (f2-f1)*r
        # nlms update g1 = g1 - lr * dg * (1/nr^2) = g1 - lr * (f2-f1) * r * (1/nr^2)
        axpy!(-lr*(f2-f1)/(nr*nr), r, g1)
        n1 = vecnorm(g1)

        if f1 < f0
            copy!(x0, x)
            copy!(g0, g)
            f0 = f1
            n0 = vecnorm(g0)
        end
        
        # Report
        cs = (n1==0) ? 0 : CUBLAS.dot(g0,g1)/(n0*n1)
        n0_avg = (1-alpha)*n0_avg + alpha * n0
        n1_avg = (1-alpha)*n1_avg + alpha * n1
        nr_avg = (1-alpha)*nr_avg + alpha * nr
        cs_avg = (1-alpha)*cs_avg + alpha * cs
        if time() >= t0
            println((nf,:f0,f0,:cos,cs_avg,:n0,n0_avg,:n1,n1_avg,:nr,nr_avg,:savg,savg,:err,wtest(x, zeroone),:lr,lr,:l2,l2,:stepsize,stepsize,:noise,noise))
            t0 = 10+time()
        end
        nf % 100 == 0 && gc()
        nf >= maxnf && break
    end
    println((nf,:f0,f0,:cos,cs_avg,:n0,n0_avg,:n1,n1_avg,:nr,nr_avg,:savg,savg,:err,wtest(x, zeroone),:lr,lr,:l2,l2,:stepsize,stepsize,:noise,noise))
    return f0
end


# Make stepsize and noise size really adaptive by experimenting and
# comparing their impact on f decrease. We try two moves from a given
# point.  The first with the best step/noise.  The second with a
# mutated step/noise.  If the second one wins we update best
# step/noise.  We use both data points to update grad estimate.

function ego5{T}(f::Function, x::BaseVector{T}; ftol=0.05, rmin=1e-6, lr=0.5, noise=2.0, step=128.0, maxnf=Inf)
    # gradient estimate
    g = fill!(similar(x), 0)

    # x0 is the best point known so far
    x0 = copy(x)
    f0 = f(x); nf = 1

    # if x2 is better than x1 we update step and noise
    # if x1 or x2 is better than x0 we update x0
    # in all cases we update the gradient estimate
    x1 = similar(x)
    x2 = similar(x)
    dx1 = similar(x)
    dx2 = similar(x)
    t0 = 0

    while f0 > ftol

        # x1 = x0 + dx1; dx1 = -step *  (g + noise  * |g| * r/|r|)  where r is a random vector
        rscale = max(rmin, vecnorm(g)/sqrt(length(x)))
        axpy!(-step, g, scale!(step * noise * rscale, randn!(dx1)))
        f1 = f(axpy!(1, dx1, copy!(x1,x0))); nf += 1

        # x2 = x0 - dx2; dx2 = step2 * (g + noise2 * |g| * r/|r|)
        step2 = step * exp(0.2*(rand()-0.5))
        noise2 = noise * exp(0.2*(rand()-0.5))
        axpy!(-step2, g, scale!(step2 * noise2 * rscale, randn!(dx2)))
        f2 = f(axpy!(1, dx2, copy!(x2,x0))); nf += 1

        # update gradient estimate:
        # h = f0 + g.dx
        # J = (h - f)^2/2
        # dg = (h - f) * dx
        # g = g - (lr/|dx|^2) * (h - f) * dx

        h1 = f0 + CUBLAS.dot(g, dx1)
        h2 = f0 + CUBLAS.dot(g, dx2)
        axpy!(-lr*(h1-f1)/vecnorm(dx1)^2, dx1, g)
        axpy!(-lr*(h2-f2)/vecnorm(dx2)^2, dx2, g)

        if f2 < f0 && f2 < f1
            copy!(x0, x2)
            f0 = f2
        end
        if f1 < f0 && f1 < f2
            copy!(x0, x1)
            f0 = f1
        end
        if f2 < f1
            step = step2
            # noise getting too low gets the algorithm stuck.
            noise2 > 1 && (noise = noise2)
        end
        if time() >= t0
            println((nf,:f0,f0,:gnorm,vecnorm(g),:step,step,:noise,noise,:lr,lr,:err,wtest(x0, zeroone)))
            t0 = 10+time()
        end
        nf % 100 == 0 && gc()
        nf >= maxnf && break
    end
    println((nf,:f0,f0,:gnorm,vecnorm(g),:step,step,:noise,noise,:lr,lr,:err,wtest(x0, zeroone)))
    return f0
end

function ego5b{T}(f::Function, x::BaseVector{T}; ftol=0.05, rmin=1e-6, lr=0.5, noise=2.0, step=128.0, maxnf=Inf)
    # gradient estimate
    g = fill!(similar(x), 0)

    # x0 is the best point known so far
    x0 = copy(x)
    f0 = f(x); nf = 1

    # if x2 is better than x1 we update step and noise
    # if x1 or x2 is better than x0 we update x0
    # in all cases we update the gradient estimate
    x1 = similar(x)
    x2 = similar(x)
    dx1 = similar(x)
    dx2 = similar(x)
    t0 = 0

    while f0 > ftol

        # x1 = x0 + dx1; dx1 = -step *  (g + noise  * |g| * r/|r|)  where r is a random vector
        rscale = max(rmin, vecnorm(g)/sqrt(length(x)))
        axpy!(-step, g, scale!(step * noise * rscale, randn!(dx1)))
        f1 = f(axpy!(1, dx1, copy!(x1,x0))); nf += 1

        if nf % 10 == 0
            # x2 = x0 - dx2; dx2 = step2 * (g + noise2 * |g| * r/|r|)
            step2 = step * exp(0.2*(rand()-0.5))
            noise2 = noise * exp(0.2*(rand()-0.5))
            axpy!(-step2, g, scale!(step2 * noise2 * rscale, randn!(dx2)))
            f2 = f(axpy!(1, dx2, copy!(x2,x0))); nf += 1
        else
            f2 = Inf
        end

        # update gradient estimate:
        # h = f0 + g.dx
        # J = (h - f)^2/2
        # dg = (h - f) * dx
        # g = g - (lr/|dx|^2) * (h - f) * dx

        h1 = f0 + CUBLAS.dot(g, dx1)
        nf%10==0 && (h2 = f0 + CUBLAS.dot(g, dx2))
        axpy!(-lr*(h1-f1)/vecnorm(dx1)^2, dx1, g)
        nf%10==0 && (axpy!(-lr*(h2-f2)/vecnorm(dx2)^2, dx2, g))

        if f2 < f0 && f2 < f1
            copy!(x0, x2)
            f0 = f2
        end
        if f1 < f0 && f1 < f2
            copy!(x0, x1)
            f0 = f1
        end
        if f2 < f1
            step = step2
            # noise getting too low gets the algorithm stuck.
            noise2 > 1 && (noise = noise2)
        end
        if time() >= t0
            println((nf,:f0,f0,:gnorm,vecnorm(g),:step,step,:noise,noise,:lr,lr,:err,wtest(x0, zeroone)))
            t0 = 10+time()
        end
        nf % 100 == 0 && gc()
        nf >= maxnf && break
    end
    println((nf,:f0,f0,:gnorm,vecnorm(g),:step,step,:noise,noise,:lr,lr,:err,wtest(x0, zeroone)))
    return f0
end

# Try separating the grad move and noise move again.

function ego6{T}(f::Function, x::BaseVector{T}; ftol=0.05, rmin=1e-6, lr=0.5, step=128.0, maxnf=Inf, salpha=0.1, smin=0.2, smax=0.8, grow=sqrt(2.0))
    # gradient estimate
    g = fill!(similar(x), 0)

    # x0 is the best point known so far
    x0 = copy(x)
    f0 = f(x); nf = 1

    # if x2 is better than x1 we update step and noise
    # if x1 or x2 is better than x0 we update x0
    # in all cases we update the gradient estimate
    x1 = similar(x)
    x2 = similar(x)
    dx = similar(x)
    r = similar(x)
    savg = 0.5
    t0 = 0

    while f0 > ftol

        # x1 = x0 - step * g
        f1 = f(axpy!(-step, g, copy!(x1,x0))); nf += 1

        # x2 = x1 + noise
        rscale = max(rmin, step*vecnorm(g)/sqrt(length(x)))
        f2 = f(axpy!(rscale, randn!(r), copy!(x2,x1))); nf += 1

        # update gradient estimate:
        # h = f0 + g.dx
        # J = (h - f)^2/2
        # dg = (h - f) * dx
        # g = g - (lr/|dx|^2) * (h - f) * dx

        axpy!(-1, x0, copy!(dx,x2))
        h2 = f0 + CUBLAS.dot(g, dx)
        axpy!(-lr*(h2-f2)/vecnorm(dx)^2, dx, g)

        # compute success rate and update stepsize accordingly
        savg = (1-salpha)*savg + salpha * (f1<f0)
        savg < smin && (step /= grow; savg = 0.5)
        savg > smax && (step *= grow; savg = 0.5)

        if f1 < f0 && f1 < f2
            f0 = f1
            copy!(x0, x1)
        elseif f2 < f0 && f2 < f1
            f0 = f2
            copy!(x0, x2)
        end

        if time() >= t0
            println((nf,:f0,f0,:gnorm,vecnorm(g),:step,step,:lr,lr,:err,wtest(x0, zeroone)))
            t0 = 10+time()
        end
        nf % 100 == 0 && gc()
        nf >= maxnf && break
    end
    println((nf,:f0,f0,:gnorm,vecnorm(g),:step,step,:lr,lr,:err,wtest(x0, zeroone)))
    return f0
end


# Trying minibatching.  Each minibatch will change f.  In order to do
# meaningful comparisons we need to compare multiple points on the
# same minibatch.  So we change the calling convention of f to take an
# arbitrary number of w's and return a corresponding number of values.

using Knet: cslice!, csize

function wmini(ws...; x=MNIST.xtrn, y=MNIST.ytrn, loss=softloss, batchsize=100)
    global xmini, ymini
    if !isdefined(:xmini) || size(xmini)!=csize(x,batchsize)
        xmini = similar(x, csize(x, batchsize))
        ymini = similar(y, csize(y, batchsize))
    end
    nd = size(x, ndims(x))
    id = rand(1:nd,batchsize)
    cslice!(xmini, x, id)
    cslice!(ymini, y, id)
    fval = map(ws) do w
        f = initmodel(w)
        ypred = forw(f, xmini)
        loss(ypred, ymini)
    end
    length(fval)==1 && (fval=fval[1])
    return fval
end

function ego7{T}(f::Function, x0::BaseVector{T}; ftol=0.05, rmin=1e-6, lr=1.0, noise=1.2, step=50.0, maxnf=Inf, maxdx=0.1)
    # gradient estimate
    g = fill!(similar(x0), 0)

    # if x1 or x2 is better than x0 we update x0
    # if x2 is better than x1 we update step and noise
    # in all cases we update the gradient estimate
    x1 = similar(x0)
    x2 = similar(x0)
    dx1 = similar(x0)
    dx2 = similar(x0)
    nf = t0 = 0; f0 = Inf

    while f0 > ftol

        # x1 = x0 + dx1; dx1 = -step *  (g + noise  * |g| * r/|r|)  where r is a random vector
        rscale = max(rmin, vecnorm(g)/sqrt(length(x0)))
        axpy!(-step, g, scale!(step * noise * rscale, randn!(dx1)))
        n1 = vecnorm(dx1); n1 > maxdx && scale!(maxdx/n1, dx1)
        axpy!(1, dx1, copy!(x1,x0))

        # x2 = x0 - dx2; dx2 = step2 * (g + noise2 * |g| * r/|r|)
        step2 = step * exp(0.2*(rand()-0.5))
        noise2 = noise * exp(0.2*(rand()-0.5))
        axpy!(-step2, g, scale!(step2 * noise2 * rscale, randn!(dx2)))
        n2 = vecnorm(dx2); n2 > maxdx && scale!(maxdx/n2, dx2)
        axpy!(1, dx2, copy!(x2,x0))

        (f0,f1,f2) = f(x0,x1,x2); nf+=3

        # update gradient estimate:
        # h = f0 + g.dx
        # J = (h - f)^2/2
        # dg = (h - f) * dx
        # g = g - (lr/|dx|^2) * (h - f) * dx

        h1 = f0 + CUBLAS.dot(g, dx1)
        h2 = f0 + CUBLAS.dot(g, dx2)
        axpy!(-lr*(h1-f1)/vecnorm(dx1)^2, dx1, g)
        axpy!(-lr*(h2-f2)/vecnorm(dx2)^2, dx2, g)

        if f2 < f0 && f2 < f1
            copy!(x0, x2)
            f0 = f2
        end
        if f1 < f0 && f1 < f2
            copy!(x0, x1)
            f0 = f1
        end
        if f2 < f1
            # step = step2
            # # noise getting too low gets the algorithm stuck.
            # noise2 > 1 && (noise = noise2)
        end
        if time() >= t0
            println((nf,:f0,f0,:gnorm,vecnorm(g),:step,step,:noise,noise,:lr,lr,:err,wtest(x0, zeroone)))
            t0 = 10+time()
        end
        nf % 100 == 0 && gc()
        nf >= maxnf && break
    end
    println((nf,:f0,f0,:gnorm,vecnorm(g),:step,step,:noise,noise,:lr,lr,:err,wtest(x0, zeroone)))
    return f0
end


# Let us simplify ego7, separating the gradient step

function ego8{T}(f::Function, x0::BaseVector{T}; ftol=0.05, rmin=1e-6, lr=1.0, noise=1.2, step=50.0, maxnf=Inf, smin=0.5, smax=0.95, alpha=0.01, grow=sqrt(2))
    g = scale!(rmin, randn!(similar(x0)))
    r = similar(g)
    x1 = similar(x0)
    nf = t0 = 0; favg = f0 = f(x0)
    savg = 0.5
    while f0 > ftol
        axpy!(-step, g, copy!(x1, x0)) # x1 = x0 - step * g
        (f0,f1) = f(x0,x1); nf+=2
        favg = (1-alpha)*favg + alpha * f0
        # compute success rate and update stepsize accordingly
        savg = (1-alpha)*savg + alpha * (f1<f0)
        #savg < smin && (step /= grow; savg = 0.5)
        #savg > smax && (step *= grow; savg = 0.5)
        if f1 < f0
            copy!(x0, x1)
            f0 = f1
        else
            # axpy!(noise*step*vecnorm(g)/sqrt(length(g)), randn!(r), x1)
            axpy!(0.1/sqrt(length(g)), randn!(r), x1)
            (f0,f1) = f(x0,x1); nf+=2
            dx = axpy!(-1, x0, x1)
            f2 = f0 + CUBLAS.dot(g, dx)
            axpy!(-lr*(f2-f1)/vecnorm(dx)^2, dx, g)
        end
        if time() >= t0
            println((nf,:favg,favg,:savg,savg,:gnorm,vecnorm(g),:step,step,:noise,noise,:lr,lr,:err,wtest(x0, zeroone)))
            t0 = 10+time()
        end
        nf % 100 == 0 && gc()
        nf >= maxnf && break
    end
    println((nf,:favg,favg,:savg,savg,:gnorm,vecnorm(g),:step,step,:noise,noise,:lr,lr,:err,wtest(x0, zeroone)))
    return wtest(x0, softloss)
end
