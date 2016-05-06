using Knet
using Knet: Net, params, axpy!
if !isdefined(:MNIST)
    include(Pkg.dir("Knet/examples/mnist.jl"))
    mnist = minibatch(MNIST.xtrn, MNIST.ytrn, 100)
end

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

# Given a minibatch (x,y), eval model and return softloss
# Note that each minibatch gives a different loss function

evalmodel(f, x, y)=softloss(forw(f,x),y)

# We can find the right epsilon by finding the max radius at which 50%
# of the steps are positive.  For mnist, this radius seems to be about
# 0.01-0.05 starting at 0.

function deneme1(r; N=100)
    global w = fill!(CudaArray(Float32,794*32),0)
    global f = initmodel(w)
    (x,y) = first(mnist)
    @show f0 = softloss(forw(f,x),y)
    w0 = copy(w)
    global dw = copy(w)
    better=0
    for i=1:N
        scale!(r, randn!(dw))
        axpy!(1, dw, w)
        f1 = softloss(forw(f,x),y)
        f1 < f0 && (better += 1)
        copy!(w, w0)
    end
    better/N
end

# Simple algorithm:
# Repeat last step at increased size if success.
# Try random step at decreased size if failure.
# 
# SGD gets to 0.36 in 1 epoch.
# This gets to around 0.60 sloss in 1000 epochs.
# using step=0.01 grow=1.001
# There is a long period where model is stuck around 0.70
#
# It turns out the grow factor is very important
# grow  sloss@64epoch
# 1.001	0.91
# 1.01	0.61
# 1.1	0.96 (though very high variance)
#
# (epoch,sloss,stepsize) using step=0.01 grow=1.01
# (epochs,step,grow) = (1024,0.01,1.01)
# (epochs,step,grow) = (1024,0.01,1.01)
# (1,1.7985317f0,0.0051531769028983964)
# (2,1.3769107f0,0.002341992948164361)
# (4,1.0358456f0,0.0011349164150302124)
# (8,0.8762574f0,0.00146262243400234)
# (16,0.7483307f0,0.0008924781269490016)
# (32,0.6375183f0,0.0008380797458788005)
# (64,0.5952995f0,0.0007391564809047644)
# (128,0.5699461f0,0.0003445230915309831)
# (256,0.59447426f0,0.0005493959444114676)
# (512,0.57121503f0,0.0002871048146276168)
# (1024,0.6157344f0,0.0008438423214864373)


function dhc01(; epochs=64, step=0.01, grow=1.01)
    @show (epochs,step,grow)
    global w = fill!(CudaArray(Float32,794*32),0)
    global f = initmodel(w)
    global dw = randn!(similar(w))
    global w0 = copy(w)
    nextn = 1
    for epoch in 1:epochs
        sloss = iter = ssize = 0
        for (x,y) in mnist
            f0 = softloss(forw(f,x),y)
            f1 = 0
            while true
                axpy!(step, dw, w)
                f1 = softloss(forw(f,x),y)
                if f1 < f0
                    sloss += f1; ssize += step; iter += 1
                    copy!(w0, w)
                    step *= grow
                    break
                else
                    copy!(w, w0)
                    step > 1e-6 && (step /= grow)
                    randn!(dw)
                end
            end
        end
        if epoch == nextn || epoch == epochs
            println((epoch,sloss/iter,ssize/iter))
            nextn*=2
        end
    end
end

# DHC1 is basically Procedure 3.1 from AITR1569.
# Let's try the original DHC algorithm, Procedure 4.1.

function dhc02(; epochs=64, step=0.01, grow=1.01, maxiter=5)
    @show (epochs,step,grow)
    global w = fill!(CudaArray(Float32,794*32),0)
    global w0 = copy(w)
    global u = fill!(similar(w), 0)
    global v = randn!(similar(w))
    global f = initmodel(w)
    nextn = 1
    for epoch in 1:epochs
        sloss = ssize = steps = 0
        for (x,y) in mnist
            f0 = softloss(forw(f,x),y)
            axpy!(step, v, w)
            f1 = softloss(forw(f,x),y)
            iter = 0
            while f1 > f0 && iter < maxiter
                iter += 1
                copy!(w, w0)
                randn!(v)
                axpy!(step, v, w)
                f1 = softloss(forw(f,x),y)
            end
            sloss += f1
            ssize += step
            steps += 1
            if f1 > f0
                step /= grow
            elseif iter == 0
                copy!(w0, w)
                axpy!(step, v, u)
                step *= grow
            else
                copy!(w0, w)    # save w+v
                axpy!(1, u, w)  # try w+v+u
                f2 = softloss(forw(f,x),y)
                if f2 > f1      # u+v did not work
                    copy!(w, w0)
                    fill!(u, 0) # reset memory
                    axpy!(step, v, u) # u is scaled v is not
                    step *= grow
                else
                    copy!(w0, w)
                    axpy!(step, v, u)
                    step *= grow
                end
            end
        end
        if epoch == nextn || epoch == epochs
            println((epoch,sloss/steps,ssize/steps))
            nextn*=2
        end
    end
end


# function train02(; epochs=1000, lr=0.5, adam=false, batchsize=100, seed=0, target=nothing, o...)
#     seed > 0 && setseed(seed)
#     global f = compile(:model02; o...)
#     global d = minibatch(MNIST.xtrn, MNIST.ytrn, batchsize)

#     # algorithm parameters
#     nhistory = 10                # remember last history steps
#     ihistory = 0
#     stepsize = 0                # dx = stepsize*gradient + (minstep + rndstep*stepsize)*random
#     minstep = 1e-4
#     rndstep = 0.1
    
#     for epoch=1:epochs
#         zloss = sloss = iter = 0
#         for (x,ygold) in d
#             ypred = forw(f, x)
#             sl = softloss(ypred, ygold)
#             zl = zeroone(ypred, ygold)
#             iter += 1; zloss += zl; sloss += sl

            

#         end
#         println((epoch, sloss/iter, zloss/iter))
#         zloss == 0 && break
#     end
#     return f
# end

# weights(f::Net)=map(r->r.out, params(f))
# gradients(f::Net)=map(r->r.dif, params(f))

# function vnorm(w)
#     sqnorm = 0
#     for u in w; sqnorm += vecnorm(u)^2; end
#     sqrt(sqnorm)
# end

# function vcos(u,v)
#     uv = 0
#     for i=1:length(u); uv += CUBLAS.dot(u[i],v[i]); end
#     uv / (vnorm(u)*vnorm(v))
# end

# function test02(f, data, loss)
#     sumloss = numloss = 0
#     for (x,ygold) in data
#         ypred = forw(f, x)
#         sumloss += loss(ypred, ygold)
#         numloss += 1
#     end
#     sumloss / numloss
# end

# f1 = train02(seed=1)
# f2 = train02(seed=1, target=f1)

# Can get to zloss=0 with hidden=24 using adam=true lr=0.001

#     setp(f, lr=lr, adam=adam)
            # # back(f, ygold, softloss)

            # if target != nothing
            #     # we start at w0 and want to reach w2
            #     # currently we are at w1 and about to take the step dw
            #     # compare dw and w2-w1
            #     # compare w2-w1 and w2-w0
            #     w1 = weights(f)
            #     dw = gradients(f)
            #     if w0 == nothing
            #         w0 = deepcopy(w1)
            #         w2 = weights(target)
            #         w2_w0 = deepcopy(w2)
            #         w2_w1 = deepcopy(w2)
            #         for i=1:length(w2); axpy!(-1,w0[i],w2_w0[i]); end
            #     end
            #     for i=1:length(w2); copy!(w2_w1[i],w2[i]); axpy!(-1,w1[i],w2_w1[i]); end
            #     @printf("w2-w0=%g,1 w2-w1=%g,%g dw=%g,%g\n",
            #             vnorm(w2_w0), vnorm(w2_w1), vcos(w2_w1, w2_w0),
            #             lr*vnorm(dw), -vcos(dw, w2_w1))
            # end

            # # update!(f)

# This doesn't work because initforw copies the arrays.
# @knet function model02(x; ftype=Float32, xinput=784, hidden=32, output=10,
#                        wvec=fill!(CudaArray(ftype,xinput*hidden+hidden*output),0))
#     h    = wf(x; f=:relu, winit=CudaArray(wvec.ptr, (hidden,xinput), wvec.dev))
#     return wf(h; f=:soft, winit=CudaArray(wvec.ptr+hidden*xinput*sizeof(ftype), (output,hidden), wvec.dev))
# end
