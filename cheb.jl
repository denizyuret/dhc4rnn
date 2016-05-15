function chebft(func::Function; a=-1.0, b=1.0, n=3)
    f = Array(Float64, n)
    c = Array(Float64, n)
    bma = 0.5*(b-a)
    bpa = 0.5*(b+a)
    for k=1:n
        y = cos(pi*(k-0.5)/n)
        f[k] = func(y*bma+bpa)
    end
    fac = 2.0/n
    for j=1:n
        sum = 0.0
        for k=1:n
            sum += f[k]*cos(pi*(j-1)*(k-0.5)/n)
        end
        c[j] = fac*sum
    end
    return c
end

function chebev(c::Vector{Float64}, x::Float64; a=-1.0, b=1.0, n=length(c))
    d = dd = 0.0
    (x-a)*(x-b) > 0 && warn("x not in range")
    y2=2.0*(y=(2.0*x-a-b)/(b-a))
    for j=n:-1:2
        sv=d
        d=y2*d-dd+c[j]
        dd=sv
    end
    return y*d-dd+0.5*c[1]
end

function chder(c::Vector{Float64}; a=-1.0, b=1.0, n=length(c))
    cder = similar(c)
    cder[n] = 0.0
    n>1 && (cder[n-1] = 2*(n-1)*c[n])
    for j=(n-2):-1:1
        cder[j] = cder[j+2]+2*j*c[j+1]
    end
    con=2.0/(b-a)
    scale!(con, cder)
end

ftmp(x)=3*x^2+5*x+8
fder(x)=6*x+5

function chebtest(f;n=5,eps=exp(-1),a=-eps,b=eps)
    f0 = chebft(f;n=n,a=a,b=b)
    f1 = chder(f0;n=n,a=a,b=b)
    f2 = chder(f1;n=n,a=a,b=b)
    y0 = chebev(f0,0.0;n=n,a=a,b=b)
    y1 = chebev(f1,0.0;n=n,a=a,b=b)
    y2 = chebev(f2,0.0;n=n,a=a,b=b)
    dx = y2>0 ? -y1/y2 : -y1/0
    df = y2>0 ? -y1*y1/y2 : -Inf
    (:err,y0-f(0),:f1,y1,:f2,y2,:dx,dx,:df,df)
end

function chebtest2(f;n=5,eps=exp(-1))
    f0 = chebft(f;n=n,a=-eps,b=eps)
    @show y0 = chebev(f0,0.0;n=n,a=-eps,b=eps)
    f1 = chder(f0;n=n,a=-eps,b=eps)
    @show y1 = chebev(f1,0.0;n=n,a=-eps,b=eps)
    @show eps = 0.1*abs(y0/y1)
    f0 = chebft(f;n=n,a=-eps,b=eps)
    @show y0 = chebev(f0,0.0;n=n,a=-eps,b=eps)
    f1 = chder(f0;n=n,a=-eps,b=eps)
    @show y1 = chebev(f1,0.0;n=n,a=-eps,b=eps)
    f2 = chder(f1;n=n,a=-eps,b=eps)
    @show y2 = chebev(f2,0.0;n=n,a=-eps,b=eps)
    (:err,y0-f(0),y1,y2)
end
