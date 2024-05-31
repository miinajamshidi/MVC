import torch
def mvc_model(nu, n, k, X, alpha, beta, landa, etha, gg):
    omega = torch.full((nu,), 1/nu)
    gamma = torch.full((nu,), 1/nu)
    torch.manual_seed(42)
    F = torch.sigmoid(torch.randn(n, k))
    Z = torch.sigmoid(torch.randn(n, n))
    H = torch.zeros((n, n))
    Zv = [torch.zeros(n, n) for _ in range(nu)]
    Fnu = [torch.rand(n, k) for _ in range(nu)]
    lap = [None] * nu  # Initialize lap with None values

    for iter in range(5):
        # calculating the L2 norm between rows and store them in H
        for i in range(n):
            for j in range(i, n):
                norm = torch.linalg.norm(F[i] - F[j], ord=2)
                H[i, j] = norm * norm
        sum = torch.zeros(n)
        gamm = 0
        sumfv = torch.zeros(n, n)
        for i in range(nu):
            u = torch.matmul(X[i].T, X[i]) + alpha[0] * torch.eye(n, n) + etha[0] * gamma[i] * torch.eye(n, n)
            inverse_u = torch.linalg.inv(u)
            gamm = gamm + gamma[i]
            sumfv = sumfv + torch.matmul(Fnu[i], Fnu[i].T)
            for j in range(n):
                t = torch.matmul(X[i].T, X[i][:, j]) - gg[0] / 4 * H[:, j] + etha[0] * gamma[i] * Z[:, j]
                zij = torch.matmul(inverse_u, t)
                Zv[i][:, j] = zij
                sum = sum + (gamma[i] * Zv[i][:, j])
        for j in range(n):
            tt = sum - (landa[0] / 4 * etha[0]) * H[:, j]
            Z[:, j] = (1 / gamm) * tt

        sumZ = torch.sum(Z, dim=1)
        DZ = torch.diag(sumZ)
        L = DZ - Z
        b = torch.zeros(n, n)
        for i in range(nu):
            Zvi = Zv[i]
            summ = torch.sum(Zvi, dim=1)
            D = torch.diag(summ)
            if lap[i] is None:  # Check if lap[i] is None
                lap[i] = D - Zvi  # Assign a value to lap[i] if it's None
            M = landa[0] * lap[i] - 2 * beta[0] * omega[i] * torch.matmul(F, F.T) - 2 * (sumfv - torch.matmul(Fnu[i], Fnu[i].T)) + beta[0] * omega[i] * torch.eye(n, n)
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            sorted_indices = torch.argsort(eigenvalues)
            sorted_eigenvalues = eigenvalues[sorted_indices]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]
            Fnu[i] = sorted_eigenvectors[:, :k]
            b = b + beta[0] * omega[i] * (2 * torch.matmul(Fnu[i], Fnu[i].T) + torch.eye(n, n))
        p = gg[0] * L - b
        eigenvalues, eigenvectors = torch.linalg.eigh(p)
        sorted_indices = torch.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        F = sorted_eigenvectors[:, :k]
        for i in range(nu):
            gamma[i] = 1 / (2 * torch.norm(Zv[i] - Z, p='fro'))
            omega[i] = 1 / (2 * torch.norm(torch.matmul(F, F.T) - torch.matmul(Fnu[i], (Fnu[i].T)), p='fro'))

    return F