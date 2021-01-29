#include <algorithm>
#include <cmath>
#include <filesystem>
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
namespace fs = std::filesystem;

bool verbose = false;

namespace Eigen {
    using MatrixXdr = Matrix<double, Dynamic, Dynamic, RowMajor>; // this is not FORTRAN
}

void print_vector(std::vector<double> const& v) {
    std::cout << "[ ";
    for(auto const& e: v) { std::cout << e << ' '; }
    std::cout << "]\n";
}

/* reads data from csv format. assumes that data consists of floating
 * point numbers only (as identified by strtof). works with header or
 * without. probably really fragile, use with caution */
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

std::vector<int> read_labels(char* fname) {/*{{{*/
    std::ifstream fin{ fname };
    if(!fin) {
        std::cerr << "invalid file name: " << fname << '\n';
        std::exit(-1);
    }

    std::vector<int> labels;
    int x;

    while(fin >> x) { labels.push_back(x); }

    std::cerr << "[read_labels] read " << labels.size() << " labels\n";
    return labels;
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

int mathematically_correct_sign(double x) {/*{{{*/
    if(x < std::numeric_limits<double>::epsilon()) { return 0; }
    return std::signbit(x) ? 1 : -1;
}/*}}}*/

Eigen::MatrixXdr compute_sq_dist_slow(Eigen::MatrixXdr const& X, Eigen::MatrixXdr const& Y) {/*{{{*/
    Eigen::MatrixXdr sq_dist{ X.rows(), Y.rows() };

    for(size_t i = 0; i < X.rows(); ++i) {
        for(size_t j = 0; j < Y.rows(); ++j) {
            sq_dist(i, j) = (X.row(i) - Y.row(j)).squaredNorm();
        }
    }

    return sq_dist;
}/*}}}*/

// binomial form: (X - Y)^2 = -2*X@Y + X^2 + Y^2
// this is faster due to optimized matrix multiplication and vectorization
// possibilities
Eigen::MatrixXdr compute_sq_dist_binomial(Eigen::MatrixXdr const& X, Eigen::MatrixXdr const& Y) {/*{{{*/
    Eigen::MatrixXd D(X.rows(), Y.rows());
    D = (
            (X * Y.transpose() * -2).colwise()
            + X.rowwise().squaredNorm()
        ).rowwise()
        + Y.rowwise().squaredNorm().transpose();
    return D;
}/*}}}*/

double tune_beta(std::vector<double> const& dist_sq_one_point, size_t const perp, double const tol=1e-5) {/*{{{*/
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
}/*}}}*/

Eigen::SparseMatrix<double> high_dimensional_affinities(std::vector<point_t> const& data, size_t perp, size_t l, size_t b, int t) {/*{{{*/
    falconn::LSHConstructionParameters params = falconn::get_default_parameters<point_t>(
        data.size(),
        data[0].size(),
        falconn::DistanceFunction::EuclideanSquared,
        true
    );

    params.l = l;
    // params.k = b; // unclear for cross polytope

    auto table = falconn::construct_table<point_t>(data, params);
    std::vector<Eigen::Triplet<double>> triplets; triplets.reserve(data.size()*3*perp);

    size_t count_not_enough = 0;

    // find k nearest neighbors for every point. k is equal to 3*perp as per the paper
    for(size_t i = 0; i < data.size(); ++i) {
        std::vector<int32_t> result; result.reserve(3*perp + 1);
        auto query = table->construct_query_object(/*num_probes*/ t != -1 ? params.l + t : t, /*max_num_candidates*/ -1);

        query->find_k_nearest_neighbors(data[i], 3*perp + 1, &result);
        result.erase(std::remove(result.begin(), result.end(), i)); // remove self from neighbors

        if(result.size() != 3*perp) {
            count_not_enough += 1;
            if(verbose) {
                std::cerr << __func__ << " [WARN] LSH query returned " << result.size() << " points instead of the requested " << 3*perp << '\n'
                          << "consider enabling multiprobing or decreasing the perplexity.\n";
            }
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

    if(count_not_enough) {
        std::cerr << __func__ << " [INFO] not enough neighbors were returned for " << count_not_enough << " points.\n"
                  << "consider enabling multiprobing or decreasing the perplexity.\n";
    }

    Eigen::SparseMatrix<double> P_j_given_i(data.size(), data.size());
    P_j_given_i.setFromTriplets(triplets.begin(), triplets.end());

    return P_j_given_i;
}/*}}}*/

template <class RNG>
void initialize_gaussian(Eigen::MatrixXdr& A, double sigma, RNG&& gen) {
    std::normal_distribution<double> dist{ 0, sigma };
    for(size_t i = 0; i < A.size(); ++i) { *(A.data() + i) = dist(gen); }
}

void initialize_PCA(Eigen::MatrixXdr& A, std::vector<point_t> const& data) {
    Eigen::MatrixXdr X(data.size(), data[0].size());
    for(size_t i = 0; i < data.size(); ++i) { X.row(i) = data[i]; }

    Eigen::MatrixXdr X_centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXdr X_cov = X_centered.adjoint() * X_centered;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXdr> eig(X_cov);

    size_t dim = A.cols();
    A = X * eig.eigenvectors().rightCols(dim);
}

/* kmeans++ initialization strategy
 *   (1) choose first centroid at random
 *   (2) for each point x, calculate the distance to the nearest previously chosen
 *       centroid D(x)
 *   (3) select the next centroid such that the probability to choose some centroid
 *       x is proportional to D(x)^2
 *   (4) repeat steps 2 and 3 until k centroids have been selected */
template <class RNG>
Eigen::MatrixXdr kmeanspp_initialize(Eigen::MatrixXdr const& Y, size_t const k, RNG&& gen) {/*{{{*/
    std::unordered_set<size_t> centroid_idxs;
    std::uniform_int_distribution<size_t> dist{ 0, static_cast<size_t>(Y.rows() - 1)};
    centroid_idxs.insert(dist(gen));

    while(centroid_idxs.size() != k) {
        std::vector<double> dist_sq; dist_sq.reserve(centroid_idxs.size());

        for(size_t i = 0; i < Y.rows(); ++i) {
            double smallest_dist_sq = INFINITY;

            for(size_t centroid_idx: centroid_idxs) {
                double centroid_dist_sq = (Y.row(centroid_idx) - Y.row(i)).squaredNorm();
                if(centroid_dist_sq < smallest_dist_sq) { smallest_dist_sq = centroid_dist_sq; }
            }

            assert(smallest_dist_sq != INFINITY);
            dist_sq.push_back(smallest_dist_sq);
        }

        std::discrete_distribution<size_t> centroid_dist{ dist_sq.begin(), dist_sq.end() };
        centroid_idxs.insert(centroid_dist(gen));
    }

    Eigen::MatrixXdr centroids{ k, Y.cols() };
    auto it = centroid_idxs.begin();
    for(auto i = 0; i < k; ++i, ++it) { centroids.row(i) = Y.row(*it); }

    return centroids;
}/*}}}*/

template <class RNG>
Eigen::MatrixXdr kmeans_initialize_random(Eigen::MatrixXdr const& Y, size_t const k, RNG&& gen) {/*{{{*/
    std::unordered_set<size_t> centroid_idxs;
    std::uniform_int_distribution<size_t> dist{ 0, static_cast<size_t>(Y.rows() - 1)};

    while(centroid_idxs.size() != k) { centroid_idxs.insert(dist(gen)); }

    Eigen::MatrixXdr centroids{ k, Y.cols() };
    auto it = centroid_idxs.begin();
    for(auto i = 0; i < k; ++i, ++it) { centroids.row(i) = Y.row(*it); }

    return centroids;
}/*}}}*/

// std::unordered_multimap can erase() an entire key, an iterator to a value or
// a range, but not a specific value. so here is a helper to find an value and
// erase it.
template <class K, class V>
void multimap_remove_single_value(std::unordered_multimap<K, V>& m, K const& key, V const& value) {/*{{{*/
    auto range = m.equal_range(key);
    auto it = range.first;

    for(/**/; it != range.second; ++it) {
        if(std::equal_to<V>{}(value, it->second)) { break; }
    }

    assert(it != range.second);
    m.erase(it);
}/*}}}*/

template <class RNG>
std::tuple<Eigen::MatrixXdr, Eigen::VectorXd, std::vector<size_t>> do_kmeans(Eigen::MatrixXdr const& Y, size_t const k, size_t const max_iter, RNG&& gen) {/*{{{*/
    std::unordered_set<size_t> centroid_idxs;
    std::uniform_int_distribution<size_t> dist{ 0, static_cast<size_t>(Y.rows() - 1)};
    while(centroid_idxs.size() != k) { centroid_idxs.insert(dist(gen)); }

    Eigen::MatrixXdr centroids = kmeans_initialize_random(Y, k, gen);
    // Eigen::MatrixXdr centroids = kmeanspp_initialize(Y, k, gen);

    std::unordered_multimap<size_t, int> cluster_assignments; // map: centroid idx -> {point idxs}
    for(size_t it = 0; it < max_iter; ++it) {
        cluster_assignments.clear();

        // assign
        Eigen::MatrixXdr sq_dist = compute_sq_dist_binomial(Y, centroids);
        for(int i = 0; i < Y.rows(); ++i) {
            Eigen::MatrixXdr::Index nearest_centroid_idx = -1;
            sq_dist.row(i).minCoeff(&nearest_centroid_idx);
            cluster_assignments.insert(std::make_pair(nearest_centroid_idx, i));
        }

        // for every cluster that is empty, find a replacement point which is
        // farthest away from its assigned centroid. more than one cluster can
        // be empty, so we make sure that a point is not chosen as a replacement
        // more than once. this implies an ordering to the replacements
        for(size_t n = 0; n < k; ++n) {
            std::unordered_set<size_t> replacement_points;

            if(cluster_assignments.count(n) == 0) {
                double farthest_dist =  0;
                size_t taken_from    = -1;
                int farthest_idx     = -1;

                for(size_t nn = 0; nn < k; ++nn) {
                    if(n == nn || cluster_assignments.count(nn) < 2) { continue; }

                    auto range = cluster_assignments.equal_range(nn);
                    for(auto it = range.first; it != range.second; ++it) {
                        double dist = (centroids.row(nn) - Y.row(it->second)).squaredNorm();

                        if(dist > farthest_dist && !replacement_points.count(it->second)) {
                            farthest_dist = dist;
                            farthest_idx  = it->second;
                            taken_from    = nn;
                        }
                    }
                }

                replacement_points.insert(farthest_idx);
                multimap_remove_single_value(cluster_assignments, taken_from, farthest_idx);

                cluster_assignments.insert(std::make_pair(n, farthest_idx));
            }
        }

        // update
        for(size_t n = 0; n < k; ++n) {
            Eigen::VectorXd new_centroid = Eigen::VectorXd::Zero(Y.cols());
            auto range = cluster_assignments.equal_range(n);
            for(auto it = range.first; it != range.second; ++it) { new_centroid += Y.row(it->second); }

            new_centroid /= cluster_assignments.count(n);
            centroids.row(n) = new_centroid;
        }

        assert(!centroids.hasNaN());
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
}/*}}}*/

double momentum(size_t iteration) { return iteration < 250 ? 0.5 : 0.8; }

void load_matrix(Eigen::MatrixXdr& X, std::string const& fname) {/*{{{*/
    std::ifstream fin{ fname };
    if(!fin) {
        std::cerr << "could not open " << fname << '\n';
        std::abort();
    }

    double x;
    for(size_t i = 0; i < X.rows(); ++i) {
        for(size_t j = 0; j < X.cols(); ++j) {
            if(!fin) { std::cerr << "fUCK\n"; }

            fin >> x;
            X(i, j) = x;
        }
    }
}/*}}}*/

void print_csv(std::string filename, Eigen::MatrixXdr const& Y, std::vector<int> labels={}) {/*{{{*/
    std::ofstream fout{ filename };

    if(!fout) {
        std::cerr << "could not open outfile " << filename << "!\n";
        std::abort();
    }

    fout << "x,y" << (labels.size() ? ",label" : "") << '\n';
    for(size_t i = 0; i < Y.rows(); ++i) {
        for(size_t j = 0; j < Y.cols(); ++j) {
            fout << Y(i, j) << (j == Y.cols()-1 ? "" : ",");
        }

        if(labels.size()) { fout << "," << labels[i]; }
        fout << '\n';
    }
}/*}}}*/

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
#define T -1
#endif

#ifndef A
#define A 5
#endif

#ifndef Z
#define Z 50
#endif

int main(int argc, char** argv) {
    size_t p = P;
    int l = L;
    int b = B;
    int t = T;
    size_t a = A;
    size_t z = Z;
    size_t s = 666;
    size_t max_iter = 250;

    double eta = 20;

    bool compute_objective = false;
    bool print_intermediate = false;

    char c;
    while((c = getopt(argc, argv, "p:l:b:t:i:n:a:z:s:vhog")) != -1) {
        switch(c) {
            case 'p':
                p = std::atoll(optarg);
                std::cerr << "p = " << p << '\n';
                break;
            case 'l':
                l = std::atoll(optarg);
                std::cerr << "l = " << l << '\n';
                break;
            case 'b':
                b = std::atoll(optarg);
                std::cerr << "b = " << b << '\n';
                break;
            case 't':
                t = std::atoll(optarg);
                std::cerr << "t = " << t << '\n';
                break;
            case 'i':
                max_iter = std::atoll(optarg);
                std::cerr << "# of iterations = " << max_iter << '\n';
                break;
            case 'o':
                compute_objective = true;
                std::cerr << "computing objective on\n";
                break;
            case 'n':
                eta = std::atof(optarg);
                std::cerr << "eta = " << eta << '\n';
                break;
            case 'v':
                verbose = true;
                std::cerr << "verbose mode on\n";
                break;
            case 'a':
                a = std::atoll(optarg);
                std::cerr << "lower bound for kmeans k a = " << a << '\n';
                break;
            case 'z':
                a = std::atoll(optarg);
                std::cerr << "upper bound for kmeans k z = " << z << '\n';
                break;
            case 's':
                s = std::atoll(optarg);
                std::cerr << "seed = " << s << '\n';
                break;
            case 'g':
                print_intermediate = true;
                std::cerr << "printing intermediate embedding to file\n";
                break;
            case 'h':
            default:
                std::cerr << "usage: " << argv[0] << " opts FILE\n"
                          << "where opt in opts is one of the following:\n\n"

                          << "  -p ... perplexity (effective number of neighbors per point). tunable parameter, default = " << P << '\n'
                          << "  -l ... number of hash tables for FALCONN lsh. tunable parameter, default = " << L << '\n'
                          << "  -b ... number of hash bits, controls number of buckets per table. automatically set to log2(n) if -1 is passed, default = " << B << '\n'
                          << "  -t ... number of probes for multi-probe LSH. tunable parameter (inverse relation to L), default = " << T << '\n'
                          << "  -i ... number of gradient descent iterations\n"
                          << "  -o ... compute objective in every iteration, default = false\n"
                          << "  -n ... set eta, default = " << eta << '\n'
                          << "  -a ... lower bound for k-means k, default = " << A << '\n'
                          << "  -z ... upper bound for k-means k, default = " << Z << '\n'
                          << "  -v ... run in verbose mode\n"
                          << "  -g ... print intermediate embedding to file every iteration (for creating GIFs)\n"
                          << "  -h ... this message\n\n"

                          << "ktsne is an accelerated approximative version of tsne which uses LSH and kmeans in its computation.\n";
                std::exit(-1);
        }
    }

    std::mt19937 gen{ s };
    std::vector<point_t> data = read_data(argv[optind]);
    std::vector<int> labels;

    if(argc > optind + 1) {
        labels = read_labels(argv[optind + 1]);

        if(labels.size() != data.size()) {
            std::cerr << "[ERR] label size mismatch. # of labels = " << labels.size() << ", # of data points = " << data.size() << "!\n";
            std::abort();
        }
    }

    normalize(data);
    center(data);

    size_t const n = data.size();
    size_t const d = 2;

    if(b == -1) { b = std::max(16ul, static_cast<size_t>(std::log2(n))); }

    if(verbose) { std::cerr << "[verbose] computing high_dimensional_affinities\n"; }
    Eigen::SparseMatrix<double> P_j_given_i = high_dimensional_affinities(data, p, l, b, t);

    Eigen::SparseMatrix<double> P_ij = Eigen::SparseMatrix<double>(P_j_given_i.transpose()) + P_j_given_i;
    P_ij /= P_ij.sum();

    Eigen::MatrixXdr Y(n, d);
    // initialize_gaussian(Y, 10e-4, gen);
    initialize_PCA(Y, data);

    Eigen::MatrixXdr iY    = Eigen::MatrixXdr::Zero(n, d);
    Eigen::MatrixXdr dY    = Eigen::MatrixXdr::Zero(n, d);
    Eigen::MatrixXdr gains = Eigen::MatrixXdr::Ones(n, d);

    std::uniform_int_distribution<size_t> k_dist{ a, z };

    if(verbose) { std::cerr << "[verbose] starting gradient descent\n"; }
    if(compute_objective) { std::cout << "it,obj,normdY\n"; }

    for(size_t it = 0; it < max_iter; ++it) {
        Eigen::MatrixXdr F_attr = Eigen::MatrixXdr::Zero(n, d);

        double early_exaggeration = it < 50 ? 12 : 1; // artificially inflate P_ij value for first few iterations

        for(int k = 0; k < P_ij.outerSize(); ++k) {
            for(Eigen::SparseMatrix<double>::InnerIterator it{ P_ij, k }; it; ++it) {
                int i = it.row(), j = it.col();
                assert(i != j);

                auto diff = Y.row(i) - Y.row(j);
                F_attr.row(i) += early_exaggeration*it.value() * (1/(1 + diff.squaredNorm())) * diff;
            }
        }

        // approximate F_rep by assigning cells using kmeans
        size_t k = k_dist(gen);
        auto [centroids, n_cell, point_assignments] = do_kmeans(Y, k, 10, gen);
        assert((n_cell.array() > 0.0).all());

        Eigen::MatrixXdr sq_dist_cell = compute_sq_dist_binomial(Y, centroids);
        assert(!sq_dist_cell.hasNaN() && sq_dist_cell.allFinite());

        Eigen::MatrixXdr q_icellZ_sq = (1/(sq_dist_cell.array() + 1).square()).matrix() * n_cell.asDiagonal();
        double Z_est = ((1/(sq_dist_cell.array() + 1)).matrix() * n_cell.asDiagonal()).sum();

        Eigen::MatrixXdr F_rep = Eigen::MatrixXdr::Zero(n, d); // NOTE: actually estimating F_repZ!
        for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j < k; ++j) {
                F_rep.row(i) += q_icellZ_sq(i, j) * (Y.row(i) - centroids.row(j));
            }
        }

        F_rep /= Z_est;
        dY = F_attr - F_rep;

        for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j < d; ++j) {
                gains(i, j) = std::max(mathematically_correct_sign(iY(i, j)) == mathematically_correct_sign(dY(i, j)) ? gains(i, j)*0.8 : gains(i, j)+0.2, 0.01);
            }
        }

        iY = momentum(it)*iY - eta*(gains.cwiseProduct(dY));
        Y += iY;

        Eigen::RowVectorXd Y_mean = Y.colwise().mean();
        Y = Y.rowwise() - Y_mean;

        if(compute_objective) {
            double kl = 0;

            for(int k = 0; k < P_ij.outerSize(); ++k) {
                for(Eigen::SparseMatrix<double>::InnerIterator it{ P_ij, k }; it; ++it) {
                    int i = it.row(), j = it.col();
                    double q_icell = std::sqrt(q_icellZ_sq(i, point_assignments[j]) / Z_est);

                    kl += it.value()*std::log2(it.value() / q_icell);
                }
            }

            std::cout << it << "," << kl << "," << dY.norm() << '\n';
        }

        if(print_intermediate && it % 5 == 0) {
            fs::create_directory("gif");
            std::string outname = std::string{ "gif/" } + fs::path(argv[optind]).stem().string() + "_embedding_it_" + std::to_string(it) + ".csv";
            print_csv(outname, Y, labels);
        }
    }

    std::stringstream outname_ss;
    outname_ss << "ktsne_" << fs::path(argv[optind]).stem().string() << "_embedding_eta_" << eta << "_p_" << p << "_s_" << s << ".csv";

    print_csv(outname_ss.str(), Y, labels);
}
