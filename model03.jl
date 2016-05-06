using Knet, Optim, CUDArt, Base.LinAlg

nrosenbrock=0

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

function rosenbrock_hessian!(x::Vector, storage::Matrix)
    storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    storage[1, 2] = -400.0 * x[1]
    storage[2, 1] = -400.0 * x[1]
    storage[2, 2] = 200.0
end

function rosenbrock_and_gradient!(x::Vector, storage)
    d1 = (1.0 - x[1])
    d2 = (x[2] - x[1]^2)
    storage[1] = -2.0 * d1 - 400.0 * d2 * x[1]
    storage[2] = 200.0 * d2
    return d1^2 + 100.0 * d2^2
end

# f = rosenbrock
# g! = rosenbrock_gradient!
# h! = rosenbrock_hessian!

function dhc(f::Function, x::Vector;                            # Procedure 4.1 from my ms thesis
             step=0.01, threshold=1e-8, maxiter=5) 
    u = fill!(similar(x), 0)
    v = randn!(similar(x))                                      # we'll use step*v instead of v in 4.1, also v=0 would not have worked in 4.1
    x0 = copy(x)
    f0 = f(x); nf = np = 1
    while step >= threshold
        iter = 0
        f1 = f(axpy!(step,v,x)); nf += 1                        # x = x0+v
        while f1 > f0 && iter < maxiter                         # do minimization instead of maximization
            randn!(v)
            f1 = f(axpy!(step,v,copy!(x,x0))); nf += 1          # x = x0+v
            iter += 1
        end
        if f1 > f0
            copy!(x,x0)                                         # x = x0
            step /= 2
        elseif iter == 0                                        # x = x0+v
            copy!(x0,x)                                         # x0' = x0+v
            axpy!(step, v, u)
            step *= 2
            f0 = f1
        else                                                    # x = x0+v
            copy!(x0,x)                                         # x0' = x0+v
            f2 = f(axpy!(1,u,x)); nf += 1                       # x = x0+v+u
            if f2 < f1                                          # why not <= here?
                copy!(x0,x)                                     # x0' = x0+v+u
                axpy!(step, v, u)                               # u=u+v
                axpy!(2, u, fill!(v,0))                         # v=2u
                f0 = f2
            else
                copy!(x,x0)                                     # x = x0+v
                axpy!(step, v, fill!(u,0))                      # u = v
                scale!(2, v)                                    # v = 2v
                f0 = f1
            end
        end
        nf >= np && (println((nf,f0,step,x[1:min(3,length(x))])); np *= 2)
    end
    (nf,f0,step,x)
end

# Make it better:

function dhc1(f::Function, x::Vector;
             vscale=1.0, xtol=1e-8, maxiter=1, grow=2, ftol=1e-4) 
    u = fill!(similar(x), 0)
    v = scale!(vscale, randn!(similar(x)))                        # vscale gives size per dimension, not total norm
    x0 = copy(x)
    fx = f0 = f(x); nf = np = 1
    while f0 > ftol
        iter = 0
        f1 = f(axpy!(1,v,x)); nf += 1                           # x = x0+v
        while f1 > f0 && iter < maxiter                         # do minimization instead of maximization
            scale!(vscale, randn!(v))
            f1 = f(axpy!(1,v,copy!(x,x0))); nf += 1          # x = x0+v
            iter += 1
        end
        if f1 > f0
            copy!(x,x0)                                         # x = x0
            vscale /= grow
            scale!(vscale, randn!(v))
        elseif iter == 0                                        # x = x0+v
            copy!(x0,x)                                         # x0' = x0+v
            axpy!(1, v, u)
            scale!(grow, v)                                     # v *= grow
            vscale *= grow
            f0 = f1
        else                                                    # x = x0+v
            copy!(x0,x)                                         # x0' = x0+v
            f2 = f(axpy!(1,u,x)); nf += 1                       # x = x0+v+u
            if f2 < f1                                          # why not <= here?
                copy!(x0,x)                                     # x0' = x0+v+u
                axpy!(1, v, u)                                  # u=u+v
                axpy!(grow, u, fill!(v,0))                      # v=2u
                vscale *= grow
                f0 = f2
            else
                copy!(x,x0)                                     # x = x0+v
                axpy!(1, v, fill!(u,0))                         # u = v
                scale!(grow, v)                                 # v = 2v
                vscale *= grow
                f0 = f1
            end
        end
        nf >= np && (println((nf,f0,vscale,x[1:min(3,length(x))])); np *= 2)
        # 0 < fx-f0 < ftol && break
        fx = f0
    end
    (nf,f0,vscale,x)
end

function sphere(x::Vector)
    s = sum(x.^2)-2*x[1]+1
    (2.0 - x[1])^2 + 100.0 * (1-s)^2
end

function sphere_gradient!(x::Vector, dx::Vector)
    s = sum(x.^2)-2*x[1]+1
    # 100*2*(1-s)*(-1)*(ds/dx)
    # -200*(1-s)*2x = -400*(1-s)*xi for i>1
    # -200*(1-s)*(2x-2) = -400*(1-s)*x1+400*(1-s) for i=1
    axpy!(-400*(1-s), x, fill!(dx,0))
    dx[1] += 400*(1-s)
    dx[1] += -2*(2-x[1])
    dx
end

# f = sphere
# g! = sphere_gradient!
