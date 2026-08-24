# PCA From Scratch

This repository implements Principal Component Analysis using only basic arithmetic operations (addition, multiplication, division, square roots) on arrays. No linear algebra library, no statistics library — every step below, including eigenvalue/eigenvector extraction, is derived and computed manually.

## 1. Theory

Let $X$ be a dataset of $n$ observations and $d$ variables, with rows $x_i^T$. Define the sample mean

$$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i$$

and center the data:

$$X_c = X - \mathbf{1}_n\bar{x}^T,\qquad x_{c,i}=x_i-\bar x.$$

For a unit vector $w$ ($w^Tw=1$), the projection $z=X_cw$ has variance

$$\text{Var}(z)=\frac{1}{n}z^Tz=w^T\left(\frac{1}{n}X_c^TX_c\right)w = w^T\Sigma w,\qquad \Sigma:=\frac{1}{n}X_c^TX_c.$$

$\Sigma_{jk}=\text{Cov}(X_j,X_k)$ — the covariance matrix is the Gram matrix of the centered data, scaled by $1/n$.

PCA maximizes this variance subject to $w^Tw=1$:

$$\mathcal{L}(w,\lambda)=w^T\Sigma w-\lambda(w^Tw-1)\ \Longrightarrow\ \Sigma w=\lambda w.$$

The maximizing directions are eigenvectors of $\Sigma$; the variance captured along each is its eigenvalue. Since $\Sigma$ is symmetric and positive semi-definite, it has real eigenvalues $\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d\ge 0$ with orthonormal eigenvectors $w_1,\dots,w_d$. Sorting eigenvectors by eigenvalue and projecting onto the top $k$ gives the principal components:

$$Z = X_cW_k,\qquad W_k=[w_1,\dots,w_k].$$

$$\text{Cov}(Z)=W_k^T\Sigma W_k=\text{diag}(\lambda_1,\dots,\lambda_k)$$

$$\text{Explained variance} = \frac{\sum_{j=1}^k \lambda_j}{\text{tr}(\Sigma)}$$

## 2. Algorithm (no library calls)

Everything below operates on raw arrays and loops. No routine computes an eigendecomposition, a covariance matrix, or a mean for you — each is built from summation and multiplication.

### 2.1 Mean

```
function MEAN(X):                     # X is n rows, d columns
    for j in 0..d-1:
        s = 0
        for i in 0..n-1:
            s = s + X[i][j]
        mean[j] = s / n
    return mean
```

### 2.2 Centering

```
function CENTER(X, mean):
    for i in 0..n-1:
        for j in 0..d-1:
            Xc[i][j] = X[i][j] - mean[j]
    return Xc
```

### 2.3 Variance (per variable)

$$\text{Var}(X_j)=\frac{1}{n}\sum_{i=1}^n (x_{ij}-\bar x_j)^2=\frac1n\sum_{i=1}^n x_{c,i,j}^2$$

```
function VARIANCE(Xc):                 # Xc is n x d, already centered
    for j in 0..d-1:
        s = 0
        for i in 0..n-1:
            s = s + Xc[i][j] * Xc[i][j]
        var[j] = s / n
    return var                          # length d
```

This is the spread of a single column on its own. It says nothing about how column $j$ moves together with column $k$ — that requires pairing two columns instead of squaring one, which is exactly the next step.

### 2.4 Covariance matrix

Replace the single squared term $x_{c,i,j}\cdot x_{c,i,j}$ with the cross term $x_{c,i,j}\cdot x_{c,i,k}$ for every pair of columns:

$$\Sigma_{jk}=\frac1n\sum_{i=1}^n x_{c,i,j}\,x_{c,i,k}$$

```
function COVARIANCE(Xc):               # Xc is n x d, already centered
    for j in 0..d-1:
        for k in 0..d-1:
            s = 0
            for i in 0..n-1:
                s = s + Xc[i][j] * Xc[i][k]
            Sigma[j][k] = s / n
    return Sigma                        # d x d
```

Note $\Sigma_{jj} = \text{VARIANCE}(Xc)[j]$ — the diagonal of the covariance matrix is exactly the variance vector computed in 2.3. `VARIANCE` is the special case $k=j$ of `COVARIANCE`; the matrix generalizes it to every pair of columns at once.

### 2.5 Eigenvalues and eigenvectors — power iteration with deflation

Since $\Sigma$ is symmetric, its dominant eigenvector can be found by repeatedly multiplying a random vector by $\Sigma$ and renormalizing — this is the power method, and it requires only vector-matrix products and a norm, both built from scratch:

```
function MATVEC(Sigma, v):              # d x d matrix times d vector
    for j in 0..d-1:
        s = 0
        for k in 0..d-1:
            s = s + Sigma[j][k] * v[k]
        result[j] = s
    return result

function NORM(v):
    s = 0
    for j in 0..d-1:
        s = s + v[j] * v[j]
    return SQRT(s)                      # Newton's method, see 2.5

function POWER_ITERATION(Sigma, iterations):
    v = random vector of length d
    v = v / NORM(v)
    for t in 1..iterations:
        v = MATVEC(Sigma, v)
        v = v / NORM(v)
    lambda = v^T * MATVEC(Sigma, v)      # Rayleigh quotient, v^T v = 1
    return lambda, v
```

To get the second eigenvector, remove the first eigenvector's contribution from $\Sigma$ (deflation) and repeat:

$$\Sigma' = \Sigma - \lambda_1 w_1 w_1^T$$

```
function DEFLATE(Sigma, lambda, w):
    for j in 0..d-1:
        for k in 0..d-1:
            Sigma[j][k] = Sigma[j][k] - lambda * w[j] * w[k]
    return Sigma

function EIGEN_DECOMPOSE(Sigma, k, iterations):
    eigenvalues = []
    eigenvectors = []
    for c in 1..k:
        lambda, w = POWER_ITERATION(Sigma, iterations)
        eigenvalues.append(lambda)
        eigenvectors.append(w)
        Sigma = DEFLATE(Sigma, lambda, w)
    return eigenvalues, eigenvectors
```

Each call finds the largest remaining eigenvalue because deflation subtracts out the variance already captured along previous directions, leaving the next-largest direction dominant for power iteration.

### 2.6 Square root (needed for norm) — Newton's method, no library call

$$x_{t+1} = \frac{1}{2}\left(x_t + \frac{a}{x_t}\right) \to \sqrt{a}$$

```
function SQRT(a):
    x = a
    repeat until |x*x - a| < tolerance:
        x = 0.5 * (x + a / x)
    return x
```

### 2.7 Projection

```
function PROJECT(Xc, eigenvectors):     # eigenvectors: list of k vectors, length d
    for i in 0..n-1:
        for c in 0..k-1:
            s = 0
            for j in 0..d-1:
                s = s + Xc[i][j] * eigenvectors[c][j]
            Z[i][c] = s
    return Z                            # n x k
```

### 2.8 Full pipeline

```
function PCA(X, k, iterations):
    mean       = MEAN(X)
    Xc         = CENTER(X, mean)
    var        = VARIANCE(Xc)           # per-variable spread, diagonal of Sigma
    Sigma      = COVARIANCE(Xc)
    eigvals, eigvecs = EIGEN_DECOMPOSE(Sigma, k, iterations)
    Z          = PROJECT(Xc, eigvecs)
    explained  = SUM(eigvals) / TRACE(Sigma_original)
    return Z, eigvals, eigvecs, explained
```

## 3. Repository structure

```
.
├── README.md
├── mean.*              # 2.1
├── center.*            # 2.2
├── variance.*          # 2.3
├── covariance.*        # 2.4
├── sqrt.*              # 2.6
├── power_iteration.*   # 2.5
├── deflate.*           # 2.5
├── eigen_decompose.*   # 2.5
├── project.*           # 2.7
├── pca.*               # 2.8 — orchestrates all steps
└── tests/
    └── ...              # verify Sigma w = lambda w, Cov(Z) diagonal, etc.
```

## 4. Correctness checks to implement

- $\Sigma w_c \approx \lambda_c\, w_c$ for every extracted pair (residual $\|\Sigma w_c - \lambda_c w_c\|$ near zero).
- $w_c^T w_{c'} \approx 0$ for $c\neq c'$ (orthogonality).
- $\text{Cov}(Z)$ is diagonal, with entries matching the extracted eigenvalues.
- $\sum_c \lambda_c \le \text{tr}(\Sigma)$, with equality when $k=d$.

## 5. Suggested learning order

1. Implement `MEAN` and `CENTER`, verify the centered data averages to zero.
2. Implement `VARIANCE`, verify against the definition on a small hand-computed example.
3. Implement `COVARIANCE`, check that its diagonal matches the output of `VARIANCE`.
4. Implement `SQRT`, then `NORM`, then `MATVEC`.
5. Implement `POWER_ITERATION` alone, check it converges to a stable vector as `iterations` grows.
6. Implement `DEFLATE` and confirm a second call to `POWER_ITERATION` finds a new, orthogonal direction.
7. Assemble `PCA` and validate against the correctness checks in Section 4.
