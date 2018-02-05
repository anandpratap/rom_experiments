import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
def flux(u):
    return 0.5*u*u
def diem(r_basis):
    n, m = r_basis.shape
    phi = np.zeros(m, dtype=np.int32);
    phi[0] = np.argmax(np.abs(r_basis[:,0]))
    e = np.eye(n)
    
    P_global = np.zeros([n,m])
    UU_global = np.zeros([n,m])
    
    P_global[:,0] = e[:, phi[0]]
    UU_global[:,0] = r_basis[:,0]
    
    for i in range(1, m):
        P = P_global[:,:i]
        UU = UU_global[:,:i]
        Ul = r_basis[:,i]
        A = np.dot(P.T, UU)
        b = np.dot(P.T, Ul)
        c = np.linalg.solve(A, b)
        r = Ul - np.dot(UU, c)
        phi[i] = np.argmax(np.abs(r))
        UU_global[:,i] = Ul
        P_global[:,i] = e[:,phi[i]]
    phi = np.sort(phi)
    P = e[:,phi]
    return phi, P
def qdiem(r_basis):
    n, m = r_basis.shape
    q, r, phi = sl.qr(r_basis.T, pivoting=True)
    phi = phi[:m]
    e = np.eye(n)
    P = e[:,phi]
    return phi, P

def update_basis(basis_new, r_basis, P):
    A = np.dot(basis_new.T, r_basis)
    An = np.dot(basis_new.T, basis_new)
    Ao = np.dot(r_basis.T, r_basis)
    den = np.sqrt(np.diag(An))*np.sqrt(np.diag(Ao))
    num = np.abs(np.diag(A))
    idx_min = np.argmin(num/den)
    basis_tmp = basis_new.copy()
    basis_tmp = np.delete(basis_tmp, [idx_min], axis=1)
    P_tmp = P.copy()
    P_tmp = np.delete(P_tmp, [idx_min])

    c = np.linalg.solve(basis_tmp[P_tmp,:], basis_new[P_tmp, idx_min])
    r = np.abs(basis_new[:,idx_min] - np.dot(basis_tmp, c))
    I = np.argsort(r)[::-1]
    for i in I:
        if i in P_tmp:
            pass
        else:
            P[idx_min] = i
            break
    return P
# [~, minI] = min(abs(diag(Unew'*Uold))./(sqrt(diag(Unew'*Unew)).*sqrt(diag(Uold'*Uold))));

# % remove maxI-th vector from Unew
# Utmp = Unew;
# Utmp(:, minI) = [];
# % remove maxI-th point from P
# Ptmp = P;
# Ptmp(minI) = [];

# % find new point following standard DEIM
# c = Utmp(Ptmp, :)\Unew(Ptmp, minI);
# r = abs(Unew(:, minI) - Utmp*c); % residual
# [~, I] = sort(r, 'descend');
# curI = 1;
# while(~isempty(find(Ptmp == I(curI), 1))) % check if point is already interp
#     curI = curI + 1;
# end
# P(minI) = I(curI); % add first point that is not already an interp point


def adiem(r_basis, P, S, fS):
    rank_tol = 1e-14
    #print S.shape
    basis_s = r_basis[S, :]
    #c = np.dot(basis_s.T, fS)
#    print basis_s.shape
 #   print fS.shape
    c = np.linalg.lstsq(basis_s, fS)[0]
  #  print c.shape, basis_s.shape, fS.shape
    res = np.dot(basis_s, c) - fS
    Q, R, E = sl.qr(c, pivoting=True)
    I = np.mean(np.abs(R), 1)/np.linalg.norm(R, 'fro')**2 > rank_tol
    R_tilde = R[I,:]
#    print resE.shape
#    print R_tilde.shape
#    print R.shape
    if R_tilde.shape[0] != 0:
        RR = np.dot(R_tilde, R_tilde.T)
        resE = res[E,:]
        dmat, rmat = sl.eig(np.dot(np.dot(R_tilde, resE.T), np.dot(resE, R_tilde.T)), RR, right=True)
        idx = np.argmax(dmat)
        normBSquare = np.dot(np.dot(rmat[:,idx].T, RR), rmat[:,idx])
        beta = np.dot(Q[:,0:R_tilde.shape[0]], rmat[:, idx])
        tmp = np.dot(res, c.T)
        alpha = -1.0/normBSquare * np.dot(tmp, beta)
        basis_new = r_basis.copy()
        #print S
        basis_new[S,:] = basis_new[S,:] + np.dot(alpha.reshape([alpha.shape[0], 1]), beta.reshape([beta.shape[0], 1]).T)
        #basis_new[:,:] = r_basis[:,:]
        P_new = update_basis(basis_new, r_basis, P)
    else:
        basis_new = r_basis.copy()
        P_new = P.copy()
    return basis_new, P_new
    #print res
    #     % relative tolerance for rank truncation
# rankTol = 1e-14;

# % compute DEIM coefficients matrix and residual at sampling points
# C = Uold(S, :)\F; % DEIM coefficients
# res = Uold(S, :)*C - F; % residual at sampling points

# % reveal rank of C matrix following Lemma 3.4
# [Q, R, E] = qr(C);
# I = mean(abs(R), 2)/norm(R, 'fro')^2 > rankTol;
# RTilde = R(I, :);
# RR = RTilde*RTilde';

# % now use RTilde instead of C to solve eigenvalue problem (Lemma 3.5)
# % compute update vectors alpha and beta
# [rMat, eMat] = eig(RTilde*(res*E)'*(res*E)*RTilde', RR);
# [~, maxI] = max(diag(eMat));
# normBSquare = rMat(:, maxI)'*RR*rMat(:, maxI);
# beta = Q(:, 1:size(RTilde, 1))*rMat(:, maxI);
# alpha = -1/normBSquare*res*C'*beta;

# % apply update to basis
# Unew = Uold;
# Unew(S, :) = Unew(S, :) + alpha*beta';

# % update DEIM interpolation points
# Pnew = updateP(Unew, Uold, Pold);

        
class Burgers(object):
    def __init__(self, mu_1 = 4.0, mu_2 = 0.02, rom_mode = False, rom_basis = None, rom_diem=False, rom_diem_basis=None, rom_adiem=False, rom_qdiem=False):
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.n = 251
        self.dt = 0.05
        self.tf = 5.0

        self.x = np.zeros(self.n)
        self.xc = np.zeros(self.n-1)
        self.uc = np.zeros(self.n-1)
        self.rc = np.zeros(self.n-1)

        self.fl = np.zeros(self.n-1)
        self.fr = np.zeros(self.n-1)

        self.rom_mode = rom_mode
        self.rom_basis = rom_basis

        self.rom_diem = rom_diem
        self.rom_qdiem = rom_qdiem
        self.rom_adiem = rom_adiem
        if self.rom_mode:
            nrow, ncol = rom_basis.shape
            self.nbasis = ncol
            self.u_hat = np.zeros(self.nbasis)
        if self.rom_diem:
            r_idx, r_P = diem(rom_diem_basis)
            if self.rom_qdiem:
                r_idx, r_P = qdiem(rom_diem_basis)
            
            A = np.dot(r_P.T, rom_diem_basis)
            A = np.dot(rom_diem_basis, np.linalg.inv(A))
            self.rom_diem_fac = A
            self.rom_diem_idx = r_idx
            self.rom_diem_basis = rom_diem_basis
            
    def initialize(self):
        self.x[:] = np.linspace(0.0, 100.0, self.n)
        self.xc[:] = 0.5*(self.x[1:] + self.x[:-1])
        self.uc[:] = 1.0

        if self.rom_mode:
            self.u_hat[:] = np.dot(self.rom_basis.T, self.uc)

    def calc_residual(self, uc):
        self.fl[0] = self.mu_1
        self.fr[0] = uc[0]

        self.fl[1:] = flux(uc[:-1])
        self.fr[1:] = flux(uc[1:])

        dx = self.x[1] - self.x[0]
        self.rc[:] = -(self.fr - self.fl)/dx + 0.02*np.exp(self.mu_2*self.xc)

    def step(self):
        if self.rom_adiem and self.i > 5:
            #print self.i
            #print self.i
            s = np.random.randint(0, self.n-1, 5)
            #print self.rom_diem_idx, s
            s = np.concatenate((self.rom_diem_idx, s))
           
            #print self.rom_diem_idx
            s = np.unique(s)
        
            rc_diem = self.r_store[self.i-5:self.i, s]#.copy()
            #print rc_diem.shape
            #print rc_diem.T.shape
            #print rc_diem.shape
            basis_new, P_new = adiem(self.rom_diem_basis, self.rom_diem_idx, s, rc_diem.T)
            #print (self.rom_diem_basis - basis_new)/self.rom_diem_basis
            self.rom_diem_basis = basis_new
            self.rom_diem_idx = P_new
            P = np.eye(self.n-1)
            P = P[:,P_new]
            A = np.dot(P.T, self.rom_diem_basis)
            A = np.dot(self.rom_diem_basis, np.linalg.inv(A))
            self.rom_diem_fac = A.copy()
            
            

        if self.rom_mode:
            uc = np.dot(self.rom_basis, self.u_hat)
        else:
            uc = self.uc
        self.calc_residual(uc)
        
        if self.rom_diem:
            rc_full = self.rc.copy()
            rc_diem = rc_full[self.rom_diem_idx]
            self.rc = np.dot(self.rom_diem_fac, rc_diem)
                
        if self.rom_mode:
            self.u_hat += np.dot(self.rom_basis.T, self.rc)*self.dt
            self.uc = np.dot(self.rom_basis, self.u_hat)
        else:
            self.uc += self.rc*self.dt
        
    def run(self, tf = None):
        if tf is None:
            tf = self.tf
        nstep = int(tf/self.dt)
        self.u_store = np.zeros([nstep+1, self.n-1])
        self.r_store = np.zeros([nstep+1, self.n-1])
        t = 0.0
        i = 0
        while t < tf:
            self.i = i
            self.u_store[i,:] = self.uc[:]
            self.step()
            self.r_store[i,:] = self.rc[:]
            t += self.dt
            i += 1
            if np.isnan(self.uc).any():
                break            
if __name__ == "__main__":
    solver = Burgers()
    solver.initialize()
    solver.run()
    plt.figure()
    plt.plot(solver.u_store[::10,:].T)
    plt.show()
         
