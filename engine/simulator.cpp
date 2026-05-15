#include "greeks.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

const double STRIKE        = 585.00;
const double EXPIRY_DAYS   = 30.0;
const double RISK_FREE     = 0.015;
const double IMPLIED_VOL   = 0.25;
const int    CONTRACT_SIZE = 100;
const double MIN_DELTA_GAP = 0.01;
const double PRICE_SCALE   = 10000.0;
const int    TRADING_START = 34200;   // 09:30:00
const int    TRADING_END   = 57600;   // 16:00:00

// Trading seconds per year: 252 days × 6.5 hours × 3600 s/hr
static const double TRADING_SECS_PER_YEAR = 252.0 * 6.5 * 3600.0;

void run_simulation(const std::string& ob_file, const std::string& msg_file, double lambda) {
    std::ifstream ob(ob_file);
    std::ifstream msg(msg_file);
    if (!ob.is_open())  { std::cerr << "Cannot open orderbook file: " << ob_file  << "\n"; return; }
    if (!msg.is_open()) { std::cerr << "Cannot open message file: "   << msg_file << "\n"; return; }

    std::ofstream tick_out ("../data/processed/tick_log.csv");
    std::ofstream hedge_out("../data/processed/hedge_log.csv");
    std::ofstream sum_out  ("../data/processed/summary.csv");
    if (!tick_out.is_open() || !hedge_out.is_open() || !sum_out.is_open()) {
        std::cerr << "Cannot open output files in ../data/processed/\n"; return;
    }

    tick_out  << "timestamp,mid_price,best_bid,best_ask,spread,bid_size_1,ask_size_1,"
                 "bid_size_2,ask_size_2,straddle_delta,straddle_gamma,straddle_theta,"
                 "delta_gap_shares,T_remaining,time_since_last_hedge,gamma_pnl_step,"
                 "theta_pnl_step,net_pnl_step,cumulative_gamma_pnl,cumulative_theta_pnl,"
                 "cumulative_tcost,rolling_ofi_50,hedged,shares_traded,exec_price,"
                 "slippage_per_share,total_slippage\n";

    hedge_out << "timestamp,mid_price,spread,delta_gap_shares,shares_traded,exec_price,"
                 "slippage_per_share,total_slippage,rolling_ofi_50,hedge_number,cumulative_tcost\n";

    // --- simulation state ---
    // Sentinel -1.0 flags "not yet seen a valid row"; set from the first valid tick.
    double prev_mid        = -1.0;
    double prev_timestamp  = -1.0;
    double last_hedge_time = -1.0;
    bool   first_row       = true;

    double position_delta        = 0.0;
    double cumulative_tcost      = 0.0;
    double cumulative_gamma_pnl  = 0.0;
    double cumulative_theta_pnl  = 0.0;
    int    hedge_count           = 0;
    int    prev_bid_size_1       = 0;
    int    prev_ask_size_1       = 0;
    int    prev_bid_size_2       = 0;
    int    prev_ask_size_2       = 0;
    int    rolling_ofi           = 0;
    std::deque<int> ofi_deque;

    // --- summary accumulators ---
    long long total_events  = 0;
    long long raw_rows      = 0;
    double spread_sum       = 0.0;
    double first_hedge_time = 0.0;
    double prev_hedge_time  = 0.0;
    bool   first_hedge_seen = false;

    std::string ob_line, msg_line;

    while (std::getline(ob, ob_line) && std::getline(msg, msg_line)) {
        ++raw_rows;

        // --- parse message: timestamp + event_type ---
        std::stringstream msg_ss(msg_line);
        std::string tok;

        if (!std::getline(msg_ss, tok, ',')) continue;
        double timestamp = std::stod(tok);

        if (!std::getline(msg_ss, tok, ',')) continue;
        int event_type = std::stoi(tok);

        if (event_type == 7) continue;
        if (timestamp < TRADING_START || timestamp > TRADING_END) continue;

        // --- parse orderbook: first 8 columns ---
        std::stringstream ob_ss(ob_line);

        auto next_ll = [&](long long& out) -> bool {
            if (!std::getline(ob_ss, tok, ',')) return false;
            out = std::stoll(tok);
            return true;
        };

        long long ask1_raw, ask_size1_raw, bid1_raw, bid_size1_raw;
        long long ask2_raw, ask_size2_raw, bid2_raw, bid_size2_raw;
        if (!next_ll(ask1_raw))     continue;
        if (!next_ll(ask_size1_raw)) continue;
        if (!next_ll(bid1_raw))     continue;
        if (!next_ll(bid_size1_raw)) continue;
        if (!next_ll(ask2_raw))     continue;
        if (!next_ll(ask_size2_raw)) continue;
        if (!next_ll(bid2_raw))     continue;
        if (!next_ll(bid_size2_raw)) continue;

        double ask1 = ask1_raw / PRICE_SCALE;
        double bid1 = bid1_raw / PRICE_SCALE;
        int bid_size1 = (int)bid_size1_raw;
        int ask_size1 = (int)ask_size1_raw;
        int bid_size2 = (int)bid_size2_raw;
        int ask_size2 = (int)ask_size2_raw;

        // guard dummy / invalid prices
        if (ask1 <= 0.0 || bid1 <= 0.0 || ask1 >= 9999.0 || bid1 <= -9999.0) continue;

        double mid_price = (ask1 + bid1) / 2.0;
        double spread = ask1 - bid1;
        double half_spread = spread / 2.0;

        if (spread < 0.0 || mid_price <= 0.0) continue;

        // --- Bootstrap: initialise state from the very first valid row ---
        // dS, dt and time_since_last_hedge are all 0 on this row, so gamma
        // and theta P&L are both zero and the hedge threshold is never crossed.
        if (first_row) {
            prev_mid       = mid_price;
            prev_timestamp = timestamp;
            last_hedge_time = timestamp;
            first_row      = false;
        }

        // --- time variables ---
        double dt                    = timestamp - prev_timestamp;
        double time_since_last_hedge = timestamp - last_hedge_time;

        // Bug 4 fix: both sides of the subtraction must be in years.
        // Elapsed calendar time: seconds → calendar years (÷ 86400 × 365)
        double T_remaining = std::max(
            EXPIRY_DAYS / 365.0 - (timestamp - TRADING_START) / (86400.0 * 365.0),
            0.001);

        // --- greeks ---
        StraddleGreeks g;
        if (T_remaining <= 0.001) {
            g.delta = 0.0; g.gamma = 0.0; g.theta = 0.0;
        } else {
            g = short_straddle_greeks(mid_price, STRIKE, T_remaining, RISK_FREE, IMPLIED_VOL);
        }

        // --- delta gap ---
        double delta_gap_shares = (g.delta * CONTRACT_SIZE) + position_delta;

        // --- hedge decision ---
        // Bug 3 fix: convert time_since_last_hedge from seconds to trading years
        // so the volatility term (sigma², annualised) is dimensionally consistent.
        double dt_years   = time_since_last_hedge / TRADING_SECS_PER_YEAR;
        double gamma_risk = 0.5 * std::abs(g.gamma)
                          * mid_price * mid_price
                          * IMPLIED_VOL * IMPLIED_VOL
                          * dt_years
                          * CONTRACT_SIZE;

        double hedge_cost = std::abs(delta_gap_shares) * half_spread;

        bool should_hedge = (gamma_risk > lambda * hedge_cost)
                         && (std::abs(delta_gap_shares) > MIN_DELTA_GAP)
                         && (T_remaining > 0.001);

        // --- execute hedge ---
        int    hedged             = 0;
        int    shares_to_trade    = 0;
        double exec_price         = 0.0;
        double slippage_per_share = 0.0;
        double total_slippage     = 0.0;

        if (should_hedge) {
            shares_to_trade = (int)std::round(delta_gap_shares);

            if (shares_to_trade != 0) {
                exec_price         = (delta_gap_shares > 0.0) ? bid1 : ask1;
                slippage_per_share = std::abs(exec_price - mid_price);
                total_slippage     = slippage_per_share * std::abs(shares_to_trade);

                position_delta   -= (double)shares_to_trade;
                cumulative_tcost += total_slippage;
                last_hedge_time   = timestamp;
                ++hedge_count;
                hedged = 1;

                if (!first_hedge_seen) {
                    first_hedge_time = timestamp;
                    first_hedge_seen = true;
                }
                prev_hedge_time = timestamp;
            }
        }

        // --- P&L components ---
        // Bug 2 fix: dS = 0 on first row (prev_mid == mid_price) → no phantom gamma P&L.
        double dS             = mid_price - prev_mid;
        double gamma_pnl_step = 0.5 * g.gamma * dS * dS * CONTRACT_SIZE;
        double theta_pnl_step = g.theta * (dt / 86400.0) * CONTRACT_SIZE;
        double net_pnl_step   = gamma_pnl_step + theta_pnl_step - total_slippage;

        cumulative_gamma_pnl += gamma_pnl_step;
        cumulative_theta_pnl += theta_pnl_step;

        // --- OFI ---
        int bid_change = bid_size1 - prev_bid_size_1;
        int ask_change = ask_size1 - prev_ask_size_1;

        int ofi_single = 0;
        if (bid_change > 0) ofi_single += 1;
        if (bid_change < 0) ofi_single -= 1;
        if (ask_change < 0) ofi_single += 1;
        if (ask_change > 0) ofi_single -= 1;

        ofi_deque.push_back(ofi_single);
        if ((int)ofi_deque.size() > 50) {
            rolling_ofi -= ofi_deque.front();
            ofi_deque.pop_front();
        }
        rolling_ofi += ofi_single;

        // --- write tick_log ---
        tick_out << std::fixed
                 << std::setprecision(6) << timestamp << ","
                 << std::setprecision(4) << mid_price << ","
                 << bid1 << ","
                 << ask1 << ","
                 << spread << ","
                 << bid_size1 << ","
                 << ask_size1 << ","
                 << bid_size2 << ","
                 << ask_size2 << ","
                 << std::setprecision(6) << g.delta << ","
                 << g.gamma << ","
                 << g.theta << ","
                 << delta_gap_shares << ","
                 << std::setprecision(4) << T_remaining << ","
                 << time_since_last_hedge << ","
                 << std::setprecision(6) << gamma_pnl_step << ","
                 << theta_pnl_step << ","
                 << net_pnl_step << ","
                 << cumulative_gamma_pnl << ","
                 << cumulative_theta_pnl << ","
                 << cumulative_tcost << ","
                 << rolling_ofi << ","
                 << hedged << ","
                 << shares_to_trade << ","
                 << std::setprecision(4) << exec_price << ","
                 << slippage_per_share << ","
                 << std::setprecision(6) << total_slippage << "\n";

        // --- write hedge_log ---
        if (hedged) {
            hedge_out << std::fixed
                      << std::setprecision(6) << timestamp << ","
                      << std::setprecision(4) << mid_price << ","
                      << spread << ","
                      << std::setprecision(6) << delta_gap_shares << ","
                      << shares_to_trade << ","
                      << std::setprecision(4) << exec_price << ","
                      << slippage_per_share << ","
                      << std::setprecision(6) << total_slippage << ","
                      << rolling_ofi << ","
                      << hedge_count << ","
                      << cumulative_tcost << "\n";
        }

        // --- update state ---
        prev_mid        = mid_price;
        prev_timestamp  = timestamp;
        prev_bid_size_1 = bid_size1;
        prev_ask_size_1 = ask_size1;
        prev_bid_size_2 = bid_size2;
        prev_ask_size_2 = ask_size2;

        ++total_events;
        spread_sum += spread;

        if (raw_rows % 100000 == 0) {
            std::cout << "Processed " << raw_rows << " rows, "
                      << total_events << " valid events, "
                      << hedge_count << " hedges executed.\n";
            std::cout.flush();
        }
    }

    // --- write summary.csv ---
    double avg_spread              = (total_events > 0) ? spread_sum / (double)total_events : 0.0;
    double avg_time_between_hedges = 0.0;
    if (hedge_count > 1)
        avg_time_between_hedges = (prev_hedge_time - first_hedge_time) / (double)(hedge_count - 1);

    double net_pnl = cumulative_gamma_pnl + cumulative_theta_pnl - cumulative_tcost;

    sum_out << "total_events,total_hedges,total_gamma_pnl,total_theta_pnl,total_tcost,"
               "net_pnl,avg_spread,avg_time_between_hedges,lambda_used,implied_vol_used,strike_used\n";
    sum_out << std::fixed << std::setprecision(6)
            << total_events << ","
            << hedge_count  << ","
            << cumulative_gamma_pnl << ","
            << cumulative_theta_pnl << ","
            << cumulative_tcost << ","
            << net_pnl << ","
            << avg_spread << ","
            << avg_time_between_hedges << ","
            << lambda << ","
            << IMPLIED_VOL << ","
            << STRIKE << "\n";

    // --- final stdout summary ---
    std::cout << "\n=== Simulation Complete ===\n"
              << std::fixed << std::setprecision(2)
              << "Total rows read        : " << raw_rows     << "\n"
              << "Valid events processed : " << total_events << "\n"
              << "Hedges executed        : " << hedge_count  << "\n"
              << "Cumulative gamma P&L   : $" << cumulative_gamma_pnl  << "\n"
              << "Cumulative theta P&L   : $" << cumulative_theta_pnl  << "\n"
              << "Total transaction costs: $" << cumulative_tcost       << "\n"
              << "Net P&L                : $" << net_pnl                << "\n"
              << std::setprecision(4)
              << "Average spread         : $" << avg_spread             << "\n"
              << std::setprecision(1)
              << "Avg time between hedges: "  << avg_time_between_hedges << "s\n"
              << "Lambda used            : "  << lambda                  << "\n";
}
