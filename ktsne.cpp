#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
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
    double log_perp = std::log2(perp);

    size_t j = 0;
    while(j++ < 200) {
        for(size_t i = 0; i < dist_sq_one_point.size(); ++i) { P[i] = std::exp(-beta * dist_sq_one_point[i]); }
        double sum_P = std::accumulate(P.begin(), P.end(), std::numeric_limits<double>::min());

        double H = 0.0;
        for(size_t i = 0; i < dist_sq_one_point.size(); ++i) { H += beta * (dist_sq_one_point[i] * P[i]); }
        H = (H / sum_P) + std::log2(sum_P);

        double H_diff = H - log_perp;
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
        for(size_t j = 0; j < result.size(); ++j) { triplets.push_back({static_cast<int>(i), result[j], P[j] / sum_P}); };
    }

    Eigen::SparseMatrix<double> P_j_given_i(data.size(), data.size());
    P_j_given_i.setFromTriplets(triplets.begin(), triplets.end());

    return P_j_given_i;
}

template <class RNG>
void initialize_gaussian(Eigen::MatrixXd& A, double sigma, RNG&& gen) {
    std::normal_distribution<double> dist{ 0, sigma };
    for(size_t i = 0; i < A.size(); ++i) { *(A.data() + i) = dist(gen); }
}

template <class RNG>
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, std::vector<size_t>> do_kmeans(Eigen::MatrixXd const& Y, size_t const k, size_t const max_iter, RNG&& gen) {
    std::unordered_set<size_t> centroid_idxs;
    std::uniform_int_distribution<size_t> dist{ 0, static_cast<size_t>(Y.rows() - 1)};
    while(centroid_idxs.size() != k) { centroid_idxs.insert(dist(gen)); }

    Eigen::MatrixXd centroids{ k, Y.cols() };
    auto it = centroid_idxs.begin();
    for(auto i = 0; i < k; ++i, ++it) { centroids.row(i) = Y.row(*it); }

    std::unordered_multimap<size_t, int> cluster_assignments; // centroid idx -> point idx
    for(size_t it = 0; it < max_iter; ++it) {
        cluster_assignments.clear();

        // assign
        for(int i = 0; i < Y.rows(); ++i) {
            std::vector<double> sq_dist; sq_dist.resize(k);
            for(size_t n = 0; n < k; ++n) { sq_dist[n] = (centroids.row(n) - Y.row(i)).squaredNorm(); }

            size_t nearest_centroid_idx = std::min_element(sq_dist.begin(), sq_dist.end()) - sq_dist.begin();
            cluster_assignments.insert(std::make_pair(nearest_centroid_idx, i));
        }

        // update
        for(size_t n = 0; n < k; ++n) {
            Eigen::VectorXd new_centroid(Y.cols());
            auto range = cluster_assignments.equal_range(n);
            for(auto it = range.first; it != range.second; ++it) { new_centroid += Y.row(it->second); }
            new_centroid /= cluster_assignments.count(n);

            centroids.row(n) = new_centroid;
        }
    }

    Eigen::VectorXd num_assigned{ k };
    for(size_t n = 0; n < k; ++n) { num_assigned(n) = cluster_assignments.count(n); }

    // reverse cluster_assignments, needed to later compute the objective by selecting the appropiate
    // centroid for every point
    std::vector<size_t> point_assignments; point_assignments.resize(Y.rows());
    for(size_t n = 0; n < k; ++n) {
        auto range = cluster_assignments.equal_range(n);
        for(auto it = range.first; it != range.second; ++it) { point_assignments[it->second] = n; }
    }

    return std::make_tuple(centroids, num_assigned, point_assignments);
}

Eigen::MatrixXd compute_sq_dist(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y) {
    Eigen::MatrixXd sq_dist{ X.rows(), Y.rows() };

    for(size_t i = 0; i < X.rows(); ++i) {
        for(size_t j = 0; j < Y.rows(); ++j) {
            sq_dist(i, j) = (X.row(i) - Y.row(j)).squaredNorm();
        }
    }

    return sq_dist;
}

double momentum(size_t iteration) { return iteration < 250 ? 0.5 : 0.8; }

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
    size_t max_iter = 250;

    double eta = 200;

    bool verbose = false;
    bool compute_objective = false;

    char c;
    while((c = getopt(argc, argv, "p:l:b:t:i:vh")) != -1) {
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
            case 'i':
                max_iter = std::atoll(optarg);
                std::cout << "# of iterations = " << max_iter << '\n';
                break;
            case 'o':
                compute_objective = true;
                std::cout << "computing objective on\n";
                break;
            case 'n':
                eta = std::atof(optarg);
                std::cout << "eta = " << eta << '\n';
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
                          << "  -o ... compute objective in every iteration, default = false\n"
                          << "  -n ... set eta, default = " << eta << '\n'
                          << "  -v ... run in verbose mode\n"
                          << "  -h ... this message\n\n"

                          << "ktsne is an accelerated version of tsne which uses LSH and kmeans in its computation.\n";
                std::exit(-1);
        }
    }

    std::mt19937 gen{ 666 };
    std::vector<point_t> data = read_data(argv[optind]);

    normalize(data);
    center(data);

    size_t const n = data.size();
    size_t const d = 2;

    if(b == -1) { b = std::max(16ul, static_cast<size_t>(std::log2(n))); }
    Eigen::SparseMatrix<double> P_j_given_i = high_dimensional_affinities(data, p, l, b, t);

    Eigen::SparseMatrix<double> P_ij = Eigen::SparseMatrix<double>(P_j_given_i.transpose()) + P_j_given_i;
    P_ij /= P_ij.sum();

    Eigen::MatrixXd Y(n, d);
    initialize_gaussian(Y, 10e-4, gen);

    Eigen::MatrixXd iY(n, d);
    Eigen::MatrixXd dY(n, d);
    Eigen::MatrixXd gains = Eigen::MatrixXd::Ones(n, d);

    for(size_t it = 0; it < max_iter; ++it) {
        Eigen::MatrixXd F_attr(n, d);

        double early_exaggeration = it < 50 ? 12 : 1; // artificially inflate P_ij value for first few iterations

        for(int k = 0; k < P_ij.outerSize(); ++k) {
            for(Eigen::SparseMatrix<double>::InnerIterator it{ P_ij, k }; it; ++it) {
                int i = it.row(), j = it.col();

                auto diff = Y.row(i) - Y.row(j);
                F_attr.row(i) += early_exaggeration*it.value() * (1/(1 + diff.squaredNorm())) * diff;
            }
        }

        // approximate F_rep by assigning cells using kmeans
        size_t k = 10;
        auto [centroids, n_cell, point_assignments] = do_kmeans(Y, k, 100, gen);

        Eigen::MatrixXd sq_dist_cell = compute_sq_dist(Y, centroids);
        Eigen::MatrixXd q_icellZ_sq = (1/(sq_dist_cell.array() + 1).square()).matrix() * n_cell.asDiagonal();

        double Z_est = ((1/(sq_dist_cell.array() + 1)).matrix() * n_cell.asDiagonal()).sum();

        Eigen::MatrixXd F_rep(n, d); // NOTE: actually estimating F_repZ!
        for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j < k; ++j) {
                F_rep.row(i) += q_icellZ_sq(i, j) * (Y.row(i) - centroids.row(j));
            }
        }

        F_rep /= Z_est;
        dY = F_attr - F_rep;

        for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j < d; ++j) {
                gains(i, j) = std::signbit(iY(i, j)) == std::signbit(dY(i, j)) ? gains(i, j)*0.8 : gains(i, j)+0.2;
                gains(i, j) = std::max(gains(i, j), 0.1);
            }
        }

        iY = momentum(it)*iY - eta*(gains*dY);
        Y += iY;

        // compute objective - optional
        if(compute_objective) {
            double kl = 0;

            for(int k = 0; k < P_ij.outerSize(); ++k) {
                for(Eigen::SparseMatrix<double>::InnerIterator it{ P_ij, k }; it; ++it) {
                    int i = it.row(), j = it.col();
                    double q_icell = std::sqrt(qiZ_cell_sq(i, point_assignments[j]) / Z_est);

                    kl += P_ij(i, j)*std::log2(P_ij(i, j) / q_icell);
                }
            }

            std::cerr << "[it = " << it << "] KL(P || Q)_est = " << kl << '\n';
        }


    }
}
