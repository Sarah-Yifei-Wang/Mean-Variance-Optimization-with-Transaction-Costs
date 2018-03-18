import pandas as pd
import numpy as np
import pandas_datareader.data as web
import cvxopt as opt
import matplotlib.pyplot as plt
from cvxopt import solvers

stocks = ['AAPL', 'F', 'SPT', 'JBLU']
trading_vol = [20500000, 36400000, 682470, 6510000]
current_portfolio = [0.25, 0.25, 0.25, 0.25]
price = [171.050003, 12.58, 21.309999, 0.025]


class PortfolioOptimization(object):
    def __init__(self):
        self.num_of_simulation = 1000
        self.total_portfolio_value = 1000000
        self.stock_data = web.DataReader(stocks, data_source='yahoo', start='1/01/2000')['Adj Close'].sort_index(axis=1)
        self.monthly_return = self.calc_monthly_ret()

    def calc_monthly_ret(self):
        """
        Download stock adjusted close price from Yahoo Finance, and then calculate monthly return

        """
        monthly_returns = self.stock_data.resample('BM', how=lambda x: (x[-1] / x[0]) - 1)
        return monthly_returns

    def calc_transaction_cost_per_share(self):
        """
        Suppose transaction cost per share for each security is an inverse function of stock average trading volume in the market.
        Becuase both bid-ask spread and market impact are negatively correlated with liquidity.
        For simplicity, we assume transaction cost = c/average_trading_volume where c = 100 here

        """
        transaction_cost_per_share = [100 / x for x in trading_vol]
        return transaction_cost_per_share

    def construct_efficient_frontier(self, min_ret, num_of_ptf, linear_coeff=None, target_ret=None):
        """
        Use quadratic programming to calculate efficient frontier for a long/short portfolio by minimizing risk
        such as portfolio return is no less than required return and the sum of all security weights is equal to 1.
        If market neutral strategy is applied, then the sum of all security weights should be set as 0

        min_ret: minimum required return
        num_of_ptf: total number of portfolios
        linear_coeff: the coefficient of the linear term. If not given, it will be set as 0
        target_ret: If target return is given, the function will return the optimal portfolio allocation to achieve that targer return.
                    If target return is not given, the function will iterate through all possible target returns starting from 0.0001 and draw a hyperbola

        """
        cov = np.matrix(self.monthly_return.cov())
        num_of_securities = len(self.monthly_return.columns)
        avg_ret_matrix = np.matrix(self.monthly_return.mean()).T
        if target_ret is None:
            target_ret = np.arange(min_ret, min_ret + num_of_ptf * min_ret, min_ret).tolist()
        cov_matrix = opt.matrix(cov)
        if linear_coeff is None:
            linear_coeff = opt.matrix(np.zeros((num_of_securities, 1)))
        modified_return_matrix = opt.matrix(np.concatenate((
            -np.transpose(np.array(avg_ret_matrix)),
            -np.identity(num_of_securities)), 0))

        all_ones_matrix = opt.matrix(1.0, (1, num_of_securities))
        sum_of_all_security_weight = opt.matrix(1.0)
        opt.solvers.options['show_progress'] = True
        if len(target_ret) > 1:
            portfolio_weights = [solvers.qp(cov_matrix, linear_coeff, modified_return_matrix,
                                            opt.matrix(np.concatenate((-np.ones((1, 1)) * yy,
                                                                       np.zeros((num_of_securities, 1))), 0)),
                                            all_ones_matrix, sum_of_all_security_weight)['x'] for yy in target_ret]
        else:
            portfolio_weights = [solvers.qp(cov_matrix, linear_coeff, modified_return_matrix,
                                            opt.matrix(np.concatenate((-np.ones((1, 1)) * target_ret,
                                                                       np.zeros((num_of_securities, 1))), 0)),
                                            all_ones_matrix, sum_of_all_security_weight)['x']]
        portfolio_returns = [(np.matrix(x).T * avg_ret_matrix)[0, 0] for x in portfolio_weights]
        portfolio_stdvs = [np.sqrt(np.matrix(x).T * cov.T.dot(np.matrix(x)))[0, 0] for x in portfolio_weights]
        return portfolio_weights, portfolio_returns, portfolio_stdvs

    def simulate_possible_portfolio(self):
        """
        This function serves as an alternative method to double check the accuracy of efficient frontier.
        This uses Monte Carlo simulation method to simulate large number of random portfolios with then sum of all security weights equal to 1,
        It is expected that all simulated portfolios sit within the efficient frontier

        """
        results = np.zeros((3 + len(stocks) - 1, self.num_of_simulation))
        for i in np.arange(0, self.num_of_simulation, 1):
            cov = np.matrix(self.monthly_return.cov())
            weights = np.array(np.random.random(4))
            weights /= np.sum(weights)
            mean_return = self.monthly_return.mean()
            portfolio_return = np.sum(mean_return * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights, np.dot(cov, weights).T))
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std_dev
            for j in range(len(weights)):
                results[j + 2, i] = weights[j]

        results_frame = pd.DataFrame(results.T,
                                     columns=['ret', 'stdev', stocks[0], stocks[1], stocks[2], stocks[3]])
        return results_frame

    def calc_transaction_cost(self, target_portfolio_weight, transaction_cost_per_share):
        """
        Calculate transaction cost for given target portfolio.
        Transaction cost depends on both security liquidity and the deviation between target portfolio and current portfolio.


        target_portfolio_weight: target weight for each security
        transaction_cost_per_share: transaction cost per share for different security

        """
        current_num_shares = [self.total_portfolio_value * weight_x / price_x for weight_x, price_x in
                              zip(current_portfolio, price)]
        target_num_shares = [self.total_portfolio_value * weight_x / price_x for weight_x, price_x in
                             zip(target_portfolio_weight, price)]
        t_cost = [transaction_cost_per_share[i] * abs(current_num_shares[i] - target_num_shares[i]) for i in
                  range(len(transaction_cost_per_share))]
        return np.array(t_cost)

    def optimize_portfolio_with_t_cost(self, return_pos, efficient_weight, efficient_mu, transaction_cost_per_share):
        """
        Assume we already have an optimal portfolio weight given target return and risk level, the function optimizes portfolio taking transaction cost into consideration.
        Transaction cost can be considered as another type of risk and is added in the quadratic function to be minimized.

        return_pos: an integer representing the postion of portfolio target return list
        efficient_weight: a list of portfolio weight that sits on efficient frontier
        efficient_mu: a list of portfolio return that sits on efficient frontier
        transaction_cost_per_share: transaction cost per share for each individual security

        """
        target_return = [efficient_mu[return_pos]]
        target_weight = efficient_weight[return_pos]
        t_cost = self.calc_transaction_cost(target_weight, transaction_cost_per_share)
        t_cost_matrix = opt.matrix(t_cost)
        opt.solvers.options['show_progress'] = True
        weights_with_t_cost, returns_with_t_cost, stdvs_with_t_cost = self.construct_efficient_frontier(0.0001, 200,
                                                                                                        t_cost_matrix,
                                                                                                        target_return)
        return weights_with_t_cost, returns_with_t_cost, stdvs_with_t_cost

    def visualize_data(self, simulation=False):
        """
        Plot scatter chart to visualize mean variance portfolio optimization
        """

        transaction_cost_per_share = self.calc_transaction_cost_per_share()
        weight, returns, sigma = self.construct_efficient_frontier(0.0001, 200)

        frontier_plot = plt.scatter(sigma, returns, color='red')
        if simulation:
            simulated_portfolio = self.simulate_possible_portfolio()
            simulation = plt.scatter(simulated_portfolio.stdev, simulated_portfolio.ret, cmap='RdYlBu')
            plt.legend((frontier_plot, simulation), ('Efficient Frontier', 'Simulated Portfolios'))
        else:
            for i in range(len(returns)):
                new_weight, new_returns, new_sigma = self.optimize_portfolio_with_t_cost(i, weight, returns,
                                                                                         transaction_cost_per_share)
                portfolio_with_t_cost = plt.scatter(new_sigma, new_returns, color='blue')
            plt.legend((frontier_plot, portfolio_with_t_cost),
                       ('Without Transaction Cost Constraint', 'With Transaction Cost Constraint'))

        plt.xlabel('Standard Deviation')
        plt.ylabel('Mean')
        plt.title('Portfolio Optimization')

        plt.show()


if __name__ == '__main__':
    optimal_portfolio = PortfolioOptimization()
    optimal_portfolio.visualize_data(False)
