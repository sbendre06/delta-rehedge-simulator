// Entry point. Parses CLI args and delegates to run_simulation().
// how to use: ./hedger <orderbook_file> <message_file> [lambda]
// lambda = hedge threshold multiplier (default 1.0, frontier.py tries out eight different vals)
#include <iostream>
#include <string>

void run_simulation(const std::string& ob_file, const std::string& msg_file, double lambda);

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./hedger <orderbook_file> <message_file> [lambda]\n"
                  << "  lambda defaults to 1.0\n";
        return 1;
    }

    std::string ob_file  = argv[1];
    std::string msg_file = argv[2];
    double lambda = 1.0;

    if (argc >= 4) {
        try {
            lambda = std::stod(argv[3]);
        } catch (...) {
            std::cerr << "Invalid lambda value: " << argv[3] << "\n";
            return 1;
        }
    }

    std::cout << "=== Friction-Aware Delta Hedging Simulator ===\n"
              << "Orderbook : " << ob_file  << "\n"
              << "Messages  : " << msg_file << "\n"
              << "Lambda    : " << lambda   << "\n\n";

    run_simulation(ob_file, msg_file, lambda);
    return 0;
}
