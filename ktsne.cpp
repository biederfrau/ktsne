#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <falconn/lsh_nn_table.h>

using point_t = falconn::DenseVector<double>;

void print_vector(std::vector<double> const& v) {
    std::cout << "[ ";
    for(auto const& e: v) { std::cout << e << ' '; }
    std::cout << "]\n";
}

/* reads data from csv format. assumes that data consists of floating
 * point numbers only (as identified by strtof). works with header or
 * without. */
std::vector<point_t> read_data(char* fname, char delim=',') {/*{{{*/
    std::ifstream fin{ fname };
    if(!fin) {
        std::cerr << "invalid file name: " << fname << '\n';
        std::exit(-1);
    }

    std::vector<point_t> data;
    size_t d = -1;
    std::string line;

    while(std::getline(fin, line)) {
        if(d == -1) { d = std::count(line.begin(), line.end(), delim) + 1; }

        std::istringstream iss{ line };
        size_t i = 0;
        point_t pi(d);
        std::string token;

        while(std::getline(iss, token, delim)) {
            char* ptr = nullptr;
            double f = std::strtof(token.data(), &ptr);

            /* did not read to end, assume that token is not double and skip token. */
            if(ptr != token.data() + token.size()) { continue; }
            pi(i++) = f;
        }

        /* only append if d elements were read */
        if(i == d) { data.push_back(pi); }
    }

    std::cerr << "[read_data] read " << data.size() << " points of dimension " << d << '\n';
    return data;
}/*}}}*/

/* normalize all points to unit vector norm */
void normalize(std::vector<point_t>& data) {/*{{{*/
    for(auto& p: data) { p.normalize(); }
}/*}}}*/

/* center points around origin */
point_t center(std::vector<point_t>& data) {/*{{{*/
    point_t center = data[0];
    for(size_t i = 1; i < data.size(); ++i) { center += data[i]; }
    center /= data.size();

    for(auto& p: data) { p -= center; }
    return center;
}/*}}}*/

double tune_beta(std::vector<double> const& dist_sq_one_point, size_t const perp, double const tol=1e-5) {
    double beta = 1.0, min_beta = std::numeric_limits<double>::lowest(), max_beta = std::numeric_limits<double>::max();
    std::vector<double> P; P.resize(dist_sq_one_point.size());

    size_t j = 0;
    while(j++ < 200) {
        for(size_t i = 0; i < dist_sq_one_point.size(); ++i) { P[i] = std::exp(-beta * dist_sq_one_point[i]); }
        double sum_P = std::accumulate(P.begin(), P.end(), std::numeric_limits<double>::min());

        double H = 0.0;
        for(size_t i = 0; i < dist_sq_one_point.size(); ++i) { H += beta * (dist_sq_one_point[i] * P[i]); }
        H = (H / sum_P) + std::log2(sum_P);

        double H_diff = H - std::log2(perp);
        if(std::abs(H_diff) < tol) { break; }

        if(H_diff > 0) {
            min_beta = beta;
            if(max_beta == std::numeric_limits<double>::max()) { beta *= 2; }
            else { beta = (beta + max_beta) / 2; }
        } else {
            max_beta = beta;
            if(min_beta == std::numeric_limits<double>::lowest()) { beta /= 2; }
            else { beta = (beta + min_beta) / 2; }
        }
    }

    return beta;
}

Eigen::SparseMatrix<double> high_dimensional_affinities(std::vector<point_t> const& data, size_t perp, size_t l, size_t b, size_t t) {
    falconn::LSHConstructionParameters params = falconn::get_default_parameters<point_t>(
        data.size(),
        data[0].size(),
        falconn::DistanceFunction::EuclideanSquared,
        true
    );

    auto table = falconn::construct_table<point_t>(data, params);
    std::vector<Eigen::Triplet<double>> triplets; triplets.reserve(data.size()*3*perp);

    // find k nearest neighbors for every point. k is equal to 3*perp as per the paper
    for(size_t i = 0; i < data.size(); ++i) {
        std::vector<int32_t> result; result.reserve(3*perp);
        auto query = table->construct_query_object(/*num_probes*/ -1, /*max_num_candidates*/ -1);

        query->find_k_nearest_neighbors(data[i], 3*perp + 1, &result);
        result.erase(std::remove(result.begin(), result.end(), i)); // remove self from neighbors

        if(result.size() != 3*perp) {
            std::cerr << __func__ << " [WARN] LSH query returned " << result.size() << " points instead of the requested " << 3*perp << '\n'
                      << "consider enabling multiprobing or decreasing the perplexity.\n";
        }

        // compute neighbor distance for every neighbor point
        std::vector<double> dist_sq_one_point; dist_sq_one_point.reserve(result.size());
        for(size_t j = 0; j < result.size(); ++j) { dist_sq_one_point.push_back((data[i] - data[result[j]]).squaredNorm()); }

        // determine sigma for this point based on its neighbors
        double beta = tune_beta(dist_sq_one_point, perp);

        // recompute P
        std::vector<double> P; P.resize(dist_sq_one_point.size());
        for(size_t j = 0; j < dist_sq_one_point.size(); ++j) { P[j] = std::exp(-beta * dist_sq_one_point[j]); }
        double sum_P = std::accumulate(P.begin(), P.end(), std::numeric_limits<double>::min());

        // row normalize and add to coefficients
        for(size_t j = 0; j < result.size(); ++j) { triplets.push_back({i, result[j], P[j] / sum_P}); };
    }

    Eigen::SparseMatrix<double> P(data.size(), data.size());
    P.setFromTriplets(triplets.begin(), triplets.end());

    return P;
}

#ifndef P
#define P 5
#endif

#ifndef L
#define L 10
#endif

#ifndef B
#define B -1
#endif

#ifndef T
#define T 1
#endif

int main(int argc, char** argv) {
    size_t p = P;
    size_t l = L;
    size_t b = B;
    size_t t = t;

    bool verbose = false;

    char c;
    while((c = getopt(argc, argv, "p:l:b:t:vh")) != -1) {
        switch(c) {
            case 'p':
                p = std::atoll(optarg);
                std::cout << "p = " << p << '\n';
            case 'l':
                l = std::atoll(optarg);
                std::cout << "l = " << l << '\n';
                break;
            case 'b':
                b = std::atoll(optarg);
                std::cout << "b = " << b << '\n';
                break;
            case 't':
                t = std::atoll(optarg);
                std::cout << "t = " << t << '\n';
                break;
            case 'v':
                verbose = true;
                std::cout << "verbose mode on\n";
                break;
            case 'h':
            default:
                std::cout << "usage: " << argv[0] << " opts FILE\n"
                          << "where opt in opts is one of the following:\n\n"

                          << "  -p ... perplexity (effective number of neighbors per point). tunable parameter, default = " << P << '\n'
                          << "  -l ... number of hash tables for FALCONN lsh. tunable parameter, default = " << L << '\n'
                          << "  -b ... number of hash bits, controls number of buckets per table. automatically set to log2(n) if -1 is passed, default = " << B << '\n'
                          << "  -t ... number of probes for multi-probe LSH. tunable parameter (inverse relation to L), default = " << T << '\n'
                          << "  -v ... run in verbose mode\n"
                          << "  -h ... this message\n\n"

                          << "ktsne is an accelerated version of tsne which uses LSH and kmeans in its computation.\n";
                std::exit(-1);
        }
    }

    std::vector<point_t> data = read_data(argv[optind]);

    normalize(data);
    center(data);

    if(b == -1) { b = std::max(16ul, static_cast<size_t>(std::log2(data.size()))); }
    Eigen::SparseMatrix<double> P_j_given_i = high_dimensional_affinities(data, p, l, b, t);
}
