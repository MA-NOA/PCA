# Principal Component Analysis — Derivation

Let $X\in\mathbb{R}^{n\times d}$ denote $n$ observations of $d$ variables, with rows $x_i^T$. Define the sample mean

$$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i\in\mathbb{R}^{d},$$

and center the data by removing this mean from every observation:

$$X_c = X - \mathbf{1}_n\bar{x}^T,\qquad x_{c,i}=x_i-\bar x.$$

Centering is not optional bookkeeping — every subsequent quantity is a second moment about $\bar x$, and without this step the algebra below would mix location with dispersion.

## Variance of a projection

Consider a unit vector $w\in\mathbb{R}^d$, $w^Tw=1$, defining a linear projection $z=X_cw\in\mathbb{R}^n$, with entries $z_i=w^Tx_{c,i}$. The sample variance of this projected scalar is

$$\operatorname{Var}(z)=\frac{1}{n}\sum_{i=1}^n z_i^2=\frac{1}{n}z^Tz=\frac{1}{n}(X_cw)^T(X_cw)=w^T\left(\frac{1}{n}X_c^TX_c\right)w.$$

The matrix appearing in this expression is defined as

$$\Sigma:=\frac{1}{n}X_c^TX_c\in\mathbb{R}^{d\times d},$$

and its $(j,k)$ entry expands as

$$\Sigma_{jk}=\frac{1}{n}\sum_{i=1}^n(x_{ij}-\bar x_j)(x_{ik}-\bar x_k)=\operatorname{Cov}(X_j,X_k).$$

So $\Sigma$ is not introduced by assumption; it arises algebraically as the Gram matrix of the centered data, and it is the unique matrix satisfying $\operatorname{Var}(z)=w^T\Sigma w$ for every direction $w$. This identity is what makes $\Sigma$ indispensable: it converts the scalar problem "variance along one direction" into a quadratic form defined over all directions simultaneously.

## Constrained optimization

The objective of PCA is to choose $w$ maximizing this variance, subject to the normalization constraint that fixes the scale of $w$:

$$\max_{w\in\mathbb{R}^d}\ w^T\Sigma w\quad\text{subject to}\quad w^Tw=1.$$

Introduce the Lagrangian

$$\mathcal{L}(w,\lambda)=w^T\Sigma w-\lambda(w^Tw-1),$$

and set the gradient with respect to $w$ to zero:

$$\nabla_w\mathcal{L}=2\Sigma w-2\lambda w=0\ \Longrightarrow\ \Sigma w=\lambda w.$$

This is not a modeling choice — it is the necessary first-order condition of the constrained optimization, and it identifies $w$ as an eigenvector of $\Sigma$ with $\lambda$ its corresponding eigenvalue. Evaluating the objective at any such stationary point gives

$$w^T\Sigma w=w^T(\lambda w)=\lambda,$$

so the variance captured along an eigenvector equals its eigenvalue exactly. The maximization is therefore solved by choosing the eigenvector associated with the largest eigenvalue.

## Spectral structure

Since $\Sigma$ is symmetric ($\Sigma^T=\Sigma$) and positive semi-definite ($w^T\Sigma w=\operatorname{Var}(z)\ge0$ for all $w$), the spectral theorem guarantees a full set of real eigenvalues

$$\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d\ge0$$

with corresponding eigenvectors $w_1,\dots,w_d$ that can be chosen orthonormal, $w_j^Tw_k=\delta_{jk}$. The eigenvector $w_1$ solves the unconstrained maximization above and defines the first principal direction.

For the second direction, the same maximization is repeated under the additional orthogonality constraint $w^Tw_1=0$, restricting the search to the orthogonal complement of $w_1$. The Lagrangian condition again yields $\Sigma w=\lambda w$, now solved within that subspace by $w_2$, the eigenvector of the next largest eigenvalue — since $\Sigma$ restricted to the orthogonal complement of an eigenspace remains symmetric with the remaining eigenpairs as its spectrum. Iterating this argument by induction over $k=1,\dots,d$ produces the full ordered eigenbasis as the complete solution sequence to the successive variance-maximization problems.

## Projection and diagonalization

Collecting the top $k$ eigenvectors into $W_k=[w_1,\dots,w_k]\in\mathbb{R}^{d\times k}$ and projecting,

$$Z=X_cW_k\in\mathbb{R}^{n\times k},$$

gives the coordinates of the data in principal component space. The covariance of the transformed data follows directly from the eigen-relations:

$$\operatorname{Cov}(Z)=\frac{1}{n}Z^TZ=W_k^T\Sigma W_k=W_k^T(W_k\Lambda_k)=\Lambda_k=\operatorname{diag}(\lambda_1,\dots,\lambda_k),$$

which is diagonal by construction — the components are uncorrelated not as an assumption but as a consequence of eigenvector orthogonality.

## Explained variance

The proportion of total variance retained by the first $k$ components is

$$\frac{\sum_{j=1}^k\lambda_j}{\sum_{j=1}^d\lambda_j}=\frac{\sum_{j=1}^k\lambda_j}{\operatorname{tr}(\Sigma)},$$

using the identity $\operatorname{tr}(\Sigma)=\sum_j\lambda_j$, which follows from the invariance of the trace under the orthogonal similarity transformation $\Sigma=W\Lambda W^T$.

---

### Wikipedia

Principal component analysis (PCA) is a linear dimensionality reduction
technique with applications in exploratory data analysis, visualization
and data preprocessing.

The data are linearly transformed onto a new coordinate system such that
the directions (principal components) capturing the largest variation in
the data can be easily identified.

---

### Other

Principal components analysis (PCA) is a method for finding low-dimensional representations of a data set that retain as much of the original variation as possible. The idea is that each of the n observations lives in p-dimensional space, but not all of these dimensions are equally interesting. In PCA we look for a smaller number of dimensions that are as interesting as possible, where the concept of interesting is measured by the amount that the observations vary along each dimension. Each of the new dimensions found in PCA is a linear combination of the original p features. The hope is to use a small subset of these linear feature combinations in further analysis while retaining most of the information present in the original data.
