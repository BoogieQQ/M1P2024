import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from tqdm.notebook import tqdm
from random import random

class VPK():

    def __init__(self, mode='corr', t=1.25, tau=0):
        self.ans = {}
        self.head = Ridge() if mode == 'linear' else LinearRegression()
        self.mode = mode
        self.tau = tau
        self.t = t
        
    def _create_predictors(self, X_train, y_train):
        predictors = {}
        for i in range(self.n_features):
            regression = LinearRegression()
            regression.fit(X_train[:, i].reshape(-1, 1), y_train)
            predictors[i] = regression
        
        self.predictors = predictors

    def _get_vars(self, X_valid):
        vars = np.zeros(self.n_features)
        for i in range(self.n_features):
            predictor = self.predictors[i]
            x = X_valid[:, i].reshape(-1, 1)
            preds = predictor.predict(x)
            vars[i] = preds.var()
        self.vars = vars

    def _get_dists(self, X_valid):
        dists = np.zeros((self.n_features, self.n_features))
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                x_i = X_valid[:, i].reshape(-1, 1)
                x_j = X_valid[:, j].reshape(-1, 1)
                predictor1, predictor2 = self.predictors[i], self.predictors[j]
                pred1, pred2 = predictor1.predict(x_i), predictor2.predict(x_j)
                dists[i][j] = ((pred1 - pred2)**2).mean()
        dists += dists.T  
        self.dists = dists

    def _get_thetas(self, y_valid):
        thetas = ((-2) * self.dists * (self.vars[:, None] @ self.vars[None, :])) / ((self.vars[:, None] - self.vars[None, :])**2 - self.dists * (self.vars[:, None] + self.vars[None, :]) + 1e-32) 

        good_thetas = np.abs(thetas * (
                    (thetas > self.vars[:, None]) * (thetas < self.vars[None, :]) + \
                    ((thetas > self.vars[:, None]) * (thetas < self.vars[None, :])).T)
              )
       
        self.cor_P = good_thetas / (self._y_valid_std * 
                    ((good_thetas - self.dists * ((good_thetas - self.vars[:, None]) * 
                    (self.vars[None, :] - good_thetas) / ((self.vars[:, None] - self.vars[None, :])**2 + 1e-32)) + 1e-32))**0.5)
        
        self.cor_V = np.sqrt(self.vars) / self._y_valid_std
        
        good_thetas = good_thetas * ((self.cor_P > self.cor_V[:, None]) | (self.cor_P > self.cor_V[None, :]))
        
        self.thetas = good_thetas

    def k(self, theta, B):
        B0, B1, B2 = B
        return theta / np.sqrt(max((1 - 0.5*B1) * theta - 0.5*B2 * theta**2 - 0.5*B0, 1e-16))
    
    def create_ensemble(self, ensemble, vars, dists, max_corr_coeff):
        P = dists[ensemble][:, ensemble]
        try:
            P1 = np.linalg.inv(P)
        except Exception as e:
            # print('Can not revers dists matrix')
            return
            
        cur_vars = vars[ensemble, None]
        alpha = (cur_vars.T @ P1 @ cur_vars)
        beta = (np.sum(P1, axis=0) @ cur_vars)
        gamma = np.sum(P1)    
    
        PHI = (P1 @ cur_vars)[:, 0]
        PSI = np.sum(P1, axis=1)
        
        denominator = (alpha * gamma - beta**2)
        denominator[denominator == 0] = 1e-16
        
        GAMMA0 = ((alpha * PSI - beta * PHI) / denominator)[0]
        GAMMA1 = ((gamma * PHI - beta * PSI) / denominator)[0]
        GAMMA1[GAMMA1 == 0] = 1e-16
        
        if ((GAMMA0 < 0) * (GAMMA1 < 0)).any():
            return
        
        mask = GAMMA0 < 0
        
        thetas_interval = - GAMMA0[mask] / GAMMA1[mask]
        thetas_interval = thetas_interval[thetas_interval > 0]
        if len(thetas_interval) == 0:
            return
        theta_min = np.max(thetas_interval)

        mask = GAMMA1 < 0
        
        thetas_interval = - GAMMA0[mask] / GAMMA1[mask]
        thetas_interval = thetas_interval[thetas_interval > 0]
        if len(thetas_interval) == 0:
            return
        theta_max = np.min(thetas_interval)
        
        B0 = np.sum(GAMMA0[:, None] @ GAMMA0[None, :] * P)
        B1 = np.sum((GAMMA0[:, None] @ GAMMA1[None, :] + GAMMA1[:, None] @ GAMMA0[None, :]) * P)
        B2 = np.sum(GAMMA1[:, None] @ GAMMA1[None, :] * P)
        B = (B0, B1, B2)

        denominator = 1 - 0.5*B1
        if abs(denominator) < 1e-16:
            denominator = 1e-16
            
        theta = B0 / denominator
        
        if theta < theta_min or theta > theta_max:
            return
        if (1 - B1) * theta - B2 * theta**2 - B0 < 0:
            return
        if (1 - B1) * theta_min - B2 * theta**2 - B0 < 0:
            return

        cur_ans_corrcoef = self.k(theta, B) / self._y_valid_std

        if cur_ans_corrcoef <= self.k(theta_min, B) / self._y_valid_std:
            return
    
        if cur_ans_corrcoef <= self.k(theta_max, B) / self._y_valid_std:
            return
        # print(max_corr_coeff)
        if cur_ans_corrcoef <= self.t * max_corr_coeff:
            return
            
        #print(ensemble, cur_ans_corrcoef)
        
        c = GAMMA0 + GAMMA1 * theta
        if tuple(sorted(ensemble[:-1])) in self.ans.keys():
            del self.ans[tuple(sorted(ensemble[:-1]))]
        
        self.ans[tuple(sorted(ensemble))] = c
    
        for i in range(self.n_features):
            if i not in ensemble:
                new_ensemble = ensemble.copy()
                new_ensemble.append(i)
                self.create_ensemble(new_ensemble, vars, dists, cur_ans_corrcoef)

    def _start(self):
        s = set()
        pairs = [i for i in zip(*self.thetas.nonzero())]
        for pair in tqdm(pairs):
            list_pair = list(pair)
            if tuple(sorted(list_pair)) not in s:
                i, j = list_pair[0], list_pair[1]
                c1 = (self.vars[j] - self.thetas[i][j]) / (self.vars[j] - self.vars[i])
                max_corr_coeff = max(self.cor_P[i][j], self.cor_V[i], self.cor_V[j])
                # print(i, j, max_corr_coeff)
                self.ans[tuple(sorted(list_pair))] = [c1, 1-c1]
                for new_index in range(self.n_features):
                    if new_index in list_pair:
                        continue
                    list_three = list_pair.copy()
                    list_three.append(new_index)
                    self.create_ensemble(list_three, self.vars, self.dists, max_corr_coeff)
                
                s.add(tuple(list(pair)))


    def _get_cor_coefs(self, X_valid, y_valid):
        corr_coefs = []
        for item in self.ans.items():
            indeces, c = item
            pred = np.zeros_like(y_valid)
            for i, ind in enumerate(indeces):
                pred += self.predictors[ind].predict(X_valid[:, ind].reshape(-1, 1)) * c[i]
            corr_coefs.append(np.corrcoef(pred, y_valid)[0][1])

        self.corr_coefs = np.array(corr_coefs)

    def _fit_head(self, X_train, y_train):
        y_train_preds = self._predict_without_head(X_train)
        self.head.fit(y_train_preds, y_train)

    def set_tau(self, tau, X_train, y_train):
        self.tau = tau
        self._fit_head(X_train, y_train)
        
    def fit(self, X_train, y_train, X_valid, y_valid):
        self.n_features = X_train.shape[1]
        self._y_valid_std = y_valid.std()
        self._create_predictors(X_train, y_train)
        self._get_vars(X_valid)
        self._get_dists(X_valid)
        self._get_thetas(y_valid)
        self._start()
        self._get_cor_coefs(X_valid, y_valid)
        self._fit_head(X_train, y_train)
        
    def _predict_without_head(self, X):
        all_preds = []
        for item in self.ans.items():
            indeces, c = item
            pred = np.zeros(len(X), dtype=np.float64)
            for i, ind in enumerate(indeces):
                pred += self.predictors[ind].predict(X[:, ind].reshape(-1, 1)) * c[i]
            all_preds.append(pred)

        all_preds = np.array(all_preds)

        mask = self.corr_coefs > self.tau * np.max(self.corr_coefs)
        
        if self.mode == 'mean':
            return all_preds[mask].mean(axis=0)[:, None]
        elif self.mode == 'corr':    
            return (all_preds[mask] * 1 / (1 - self.corr_coefs[mask]**2)[:, None]).sum(axis=0)[:, None]
        elif self.mode == 'linear':
            return np.transpose(all_preds[mask])

    
    def predict(self, X):
        return self.head.predict(self._predict_without_head(X))