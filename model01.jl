using Knet
using Knet: Net, params, axpy!
isdefined(:MNIST) || include("mnist.jl")

@knet function model01(x; winit=Gaussian(0,.1), hidden=32, f1=:relu, o...)
    h    = wf(x; out=hidden, f=f1, winit=winit)
    return wf(h; out=10, f=:soft, winit=winit)
end

function train01(; epochs=1000, lr=0.5, adam=false, batchsize=100, seed=0, target=nothing, o...)
    seed > 0 && setseed(seed)
    global f = compile(:model01; o...)
    global d = minibatch(MNIST.xtrn, MNIST.ytrn, batchsize)
    setp(f, lr=lr, adam=adam)
    global w0,w1,w2,dw,w2_w0,w2_w1
    w0 = w1 = w2 = nothing
    for epoch=1:epochs
        zloss = sloss = iter = 0
        for (x,ygold) in d
            ypred = forw(f, x)
            back(f, ygold, softloss)

            if target != nothing
                # we start at w0 and want to reach w2
                # currently we are at w1 and about to take the step dw
                # compare dw and w2-w1
                # compare w2-w1 and w2-w0
                w1 = weights(f)
                dw = gradients(f)
                if w0 == nothing
                    w0 = deepcopy(w1)
                    w2 = weights(target)
                    w2_w0 = deepcopy(w2)
                    w2_w1 = deepcopy(w2)
                    for i=1:length(w2); axpy!(-1,w0[i],w2_w0[i]); end
                end
                for i=1:length(w2); copy!(w2_w1[i],w2[i]); axpy!(-1,w1[i],w2_w1[i]); end
                @printf("w2-w0=%g,1 w2-w1=%g,%g dw=%g,%g\n",
                        vnorm(w2_w0), vnorm(w2_w1), vcos(w2_w1, w2_w0),
                        lr*vnorm(dw), -vcos(dw, w2_w1))
            end

            update!(f)
            zloss += zeroone(ypred, ygold)
            sloss += softloss(ypred, ygold)
            iter += 1
        end
        println((epoch, sloss/iter, zloss/iter))
        zloss == 0 && break
    end
    return f
end

weights(f::Net)=map(r->r.out, params(f))
gradients(f::Net)=map(r->r.dif, params(f))

function vnorm(w)
    sqnorm = 0
    for u in w; sqnorm += vecnorm(u)^2; end
    sqrt(sqnorm)
end

function vcos(u,v)
    uv = 0
    for i=1:length(u); uv += CUBLAS.dot(u[i],v[i]); end
    uv / (vnorm(u)*vnorm(v))
end

function test01(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    sumloss / numloss
end

f1 = train01(seed=1)
f2 = train01(seed=1, target=f1)

# Can get to zloss=0 with hidden=24 using adam=true lr=0.001
