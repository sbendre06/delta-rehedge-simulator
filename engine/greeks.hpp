#pragma once
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

inline double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

inline double bs_d1(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0 || S <= 0.0 || K <= 0.0) return 0.0;
    return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}

inline double bs_d2(double S, double K, double T, double r, double sigma) {
    return bs_d1(S, K, T, r, sigma) - sigma * std::sqrt(T);
}

inline double bs_delta_call(double S, double K, double T, double r, double sigma) {
    return norm_cdf(bs_d1(S, K, T, r, sigma));
}

inline double bs_delta_put(double S, double K, double T, double r, double sigma) {
    return norm_cdf(bs_d1(S, K, T, r, sigma)) - 1.0;
}

inline double bs_gamma(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0 || S <= 0.0) return 0.0;
    return norm_pdf(bs_d1(S, K, T, r, sigma)) / (S * sigma * std::sqrt(T));
}

inline double bs_theta_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0 || S <= 0.0) return 0.0;
    double d1 = bs_d1(S, K, T, r, sigma);
    double d2 = bs_d2(S, K, T, r, sigma);
    return -(S * norm_pdf(d1) * sigma / (2.0 * std::sqrt(T))
             + r * K * std::exp(-r * T) * norm_cdf(d2)) / 365.0;
}

inline double bs_theta_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0 || S <= 0.0) return 0.0;
    double d1 = bs_d1(S, K, T, r, sigma);
    double d2 = bs_d2(S, K, T, r, sigma);
    return -(S * norm_pdf(d1) * sigma / (2.0 * std::sqrt(T))
             - r * K * std::exp(-r * T) * norm_cdf(-d2)) / 365.0;
}

struct StraddleGreeks {
    double delta;
    double gamma;
    double theta;
};

inline StraddleGreeks short_straddle_greeks(double S, double K, double T, double r, double sigma) {
    double call_delta    = bs_delta_call(S, K, T, r, sigma);
    double put_delta     = bs_delta_put(S, K, T, r, sigma);
    double single_gamma  = bs_gamma(S, K, T, r, sigma);
    double call_theta    = bs_theta_call(S, K, T, r, sigma);
    double put_theta     = bs_theta_put(S, K, T, r, sigma);

    StraddleGreeks g;
    g.delta = -(call_delta + put_delta);
    g.gamma = -(2.0 * single_gamma);
    g.theta = -(call_theta + put_theta);
    return g;
}
