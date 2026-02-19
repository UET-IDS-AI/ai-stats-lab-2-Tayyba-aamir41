"""
AI Mathematical Tools – Probability & Random Variables

Instructions:
- Implement ALL functions.
- Do NOT change function names or signatures.
- Do NOT print inside functions.
- You may use: math, numpy, matplotlib.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Part 1 — Probability Foundations
# ============================================================

def probability_union(PA, PB, PAB):
    result= PA + PB - PAB   #P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    return result


def conditional_probability(PAB, PB):
    result = PAB / PB   #  P(A|B) = P(A ∩ B) / P(B)
    return result


def are_independent(PA, PB, PAB, tol=1e-9):
   
    step_1 = PA * PB
    step_2=abs(PAB - step_1)  # abs mean absolute value
    return step_2< tol        # |P(A ∩ B) - P(A)P(B)| < tol


def bayes_rule(PBA, PA, PB):
    return (PBA * PA) / PB    # P(A|B) = P(B|A)P(A) / P(B)


# ============================================================
# Part 2 — Bernoulli Distribution
# ============================================================

def bernoulli_pmf(x, theta):
    step_1 = theta**x
    step_2 = (1-theta)**(1-x)
    result = step_1 * step_2
    return result             #result = theta^x (1-theta)^(1-x)


def bernoulli_theta_analysis(theta_values):
    results = []
    for theta in theta_values:
        P0 = bernoulli_pmf(0, theta)
        P1 = bernoulli_pmf(1, theta)
        is_symmetric = abs(P0-P1) < 1e-9 
        results.append((theta, P0, P1, is_symmetric))
    return results


# ============================================================
# Part 3 — Normal Distribution
# ============================================================

def normal_pdf(x, mu, sigma):
    """Normal PDF"""
    coeff = 1 / (math.sqrt(2 * math.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coeff * math.exp(exponent)

def normal_histogram_analysis(mu_values,
                              sigma_values,
                              n_samples=10000,
                              bins=30,
                              plot=False):
    """Analyze normal distribution samples"""
    results = []

    for mu, sigma in zip(mu_values, sigma_values):
        samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)
        sample_mean = np.mean(samples)
        sample_variance = np.var(samples)
        theoretical_mean = mu
        theoretical_variance = sigma ** 2
        mean_error = abs(sample_mean - theoretical_mean)
        variance_error = abs(sample_variance - theoretical_variance)

        results.append(
            (mu, sigma, sample_mean, theoretical_mean, mean_error,
             sample_variance, theoretical_variance, variance_error)
        )

        if plot:
            plt.hist(samples, bins=bins, density=True, alpha=0.6, color='g')
            plt.title(f'Normal Distribution μ={mu}, σ={sigma}')
            plt.show()

    return results

# ============================================================
# Part 4 — Uniform Distribution
# ============================================================

def uniform_mean(a, b):
    """Theoretical mean of Uniform(a, b)"""
    return (a + b) / 2

def uniform_variance(a, b):
    """Theoretical variance of Uniform(a, b)"""
    return ((b - a) ** 2) / 12

def uniform_histogram_analysis(a_values,
                               b_values,
                               n_samples=10000,
                               bins=30,
                               plot=False):
    """Analyze uniform distribution samples"""
    results = []

    for a, b in zip(a_values, b_values):
        samples = np.random.uniform(low=a, high=b, size=n_samples)
        sample_mean = np.mean(samples)
        sample_variance = np.var(samples)
        theoretical_mean = uniform_mean(a, b)
        theoretical_variance = uniform_variance(a, b)
        mean_error = abs(sample_mean - theoretical_mean)
        variance_error = abs(sample_variance - theoretical_variance)

        results.append(
            (a, b, sample_mean, theoretical_mean, mean_error,
             sample_variance, theoretical_variance, variance_error)
        )

        if plot:
            plt.hist(samples, bins=bins, density=True, alpha=0.6, color='b')
            plt.title(f'Uniform Distribution a={a}, b={b}')
            plt.show()

    return results


if __name__ == "__main__":
    print("Implement all required functions.")
