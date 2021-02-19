#include <algorithm>
#include <cmath>
#include <fstream>

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <getopt.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <falconn/lsh_nn_table.h>

using point_t = falconn::DenseVector<double>;

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

double compute_procrustes(Eigen::MatrixXdr const& X, Eigen::MatrixXdr const& Y) {
    return (X - Y).array().square().sum();
}

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

Eigen::SparseMatrix<double> high_dimensional_affinities(std::vector<point_t> const& data, size_t perp, size_t num_hash_tables, size_t bits, int num_probes, bool use_hyperplane=false) {/*{{{*/
    falconn::LSHConstructionParameters params = falconn::get_default_parameters<point_t>(
        data.size(),
        data[0].size(),
        falconn::DistanceFunction::EuclideanSquared,
        true
    );

    params.l = num_hash_tables;
    if(use_hyperplane) {
        params.lsh_family = falconn::LSHFamily::Hyperplane;
        params.k = bits; // unclear for cross polytope
    }

    auto table = falconn::construct_table<point_t>(data, params);
    std::vector<Eigen::Triplet<double>> triplets; triplets.reserve(data.size()*3*perp);

    size_t count_not_enough = 0;

    // find k nearest neighbors for every point. k is equal to 3*perp as per the paper
    for(size_t i = 0; i < data.size(); ++i) {
        std::vector<int32_t> result; result.reserve(3*perp + 1);
        auto query = table->construct_query_object(/*num_probes*/ num_probes != -1 ? params.l + num_probes : num_probes, /*max_num_candidates*/ -1);

        query->find_k_nearest_neighbors(data[i], 3*perp + 1, &result);
        result.erase(std::remove(result.begin(), result.end(), i)); // remove self from neighbors

        if(result.size() != 3*perp) { count_not_enough += 1; }

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

Eigen::MatrixXdr::Index find_min(Eigen::MatrixXdr const& mat, int row) {
    double mini = INFINITY;
    int mini_index;

    for(int j = 0; j < mat.cols(); ++j) {
        if(mat(row, j) < mini) {
            mini = mat(row, j);
            mini_index = j;
        }
    }

    return mini_index;
}

template <class RNG>
std::tuple<Eigen::MatrixXdr, Eigen::VectorXd, std::vector<size_t>> do_kmeans(Eigen::MatrixXdr const& Y, size_t const k, size_t const max_iter, RNG&& gen, bool compute_objective=false) {/*{{{*/
    std::unordered_set<size_t> centroid_idxs;
    std::uniform_int_distribution<size_t> dist{ 0, static_cast<size_t>(Y.rows() - 1)};
    while(centroid_idxs.size() != k) { centroid_idxs.insert(dist(gen)); }

    Eigen::MatrixXdr centroids = kmeans_initialize_random(Y, k, gen);
    // Eigen::MatrixXdr centroids = kmeanspp_initialize(Y, k, gen);

    std::vector<std::vector<size_t>> cluster_assignments; cluster_assignments.resize(k);
    for(size_t it = 0; it < max_iter; ++it) {
        for(auto& vec: cluster_assignments) { vec.clear(); vec.reserve(Y.rows()/k); } // clear should not change capacity(), so this should only allocate the very first time!

        // assign
        Eigen::MatrixXdr sq_dist = compute_sq_dist_binomial(Y, centroids);
        for(int i = 0; i < Y.rows(); ++i) {
            Eigen::MatrixXdr::Index nearest_centroid_idx = find_min(sq_dist, i);
            cluster_assignments[nearest_centroid_idx].push_back(i);
        }

        // for every cluster that is empty, find a replacement point which is
        // farthest away from its assigned centroid. more than one cluster can
        // be empty, so we make sure that a point is not chosen as a replacement
        // more than once. this implies an ordering to the replacements
        for(size_t n = 0; n < k; ++n) {
            std::unordered_set<size_t> replacement_points;

            if(cluster_assignments[n].size() == 0) {
                double farthest_dist =  0;
                size_t taken_from    = -1;
                int farthest_idx     = -1;

                for(size_t nn = 0; nn < k; ++nn) {
                    if(n == nn || cluster_assignments[nn].size() < 2) { continue; }

                    auto range = std::make_pair(cluster_assignments[nn].begin(), cluster_assignments[nn].end());
                    for(auto it = range.first; it != range.second; ++it) {
                        double dist = (centroids.row(nn) - Y.row(*it)).squaredNorm();

                        if(dist > farthest_dist && !replacement_points.count(*it)) {
                            farthest_dist = dist;
                            farthest_idx  = *it;
                            taken_from    = nn;
                        }
                    }
                }

                replacement_points.insert(farthest_idx);
                std::remove(cluster_assignments[taken_from].begin(), cluster_assignments[taken_from].end(), farthest_idx);

                cluster_assignments[n].push_back(farthest_idx);
            }
        }

        // update
        for(size_t n = 0; n < k; ++n) {
            Eigen::VectorXd new_centroid = Eigen::VectorXd::Zero(Y.cols());
            auto range = std::make_pair(cluster_assignments[n].begin(), cluster_assignments[n].end());
            for(auto it = range.first; it != range.second; ++it) { new_centroid += Y.row(*it); }

            new_centroid /= cluster_assignments[n].size();
            centroids.row(n) = new_centroid;
        }

        assert(!centroids.hasNaN());
    }

    Eigen::VectorXd num_assigned{ k };
    for(size_t n = 0; n < k; ++n) { num_assigned(n) = cluster_assignments[n].size(); }

    // reverse cluster_assignments, needed to later compute the objective by selecting the appropiate
    // centroid for every point
    std::vector<size_t> point_assignments;
    if(compute_objective) {
        point_assignments.resize(Y.rows());
        for(size_t n = 0; n < k; ++n) {
            auto range = std::make_pair(cluster_assignments[n].begin(), cluster_assignments[n].end());
            for(auto it = range.first; it != range.second; ++it) { point_assignments[*it] = n; }
        }
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

struct config_t {
    double eta                   = 200;
    unsigned int perplexity      = 50;
    double early_exaggeration    = 12;
    double late_exaggeration     = 4;

    unsigned int num_hash_tables = 100;
             int num_hash_bits   = -1;
             int num_probes      = -1;

    unsigned int kmeans_lo       = 5;
    unsigned int kmeans_hi       = 50;

    unsigned int seed            = 666;
    size_t max_iter              = 1000;

    int compute_objective        = 0;
    int print_intermediate       = 0;
    int use_hyperplane           = 0;
    int random_init              = 0;
};

void print_usage(std::string const& program) {
    config_t default_config;
    std::cerr << "usage: " << program << " opts FILE\n"
              << "where opt in opts is one of the following:\n\n"

              << "  -p                       ... perplexity (effective number of neighbors per point). tunable parameter, default = " << default_config.perplexity << '\n'
              << "  -n                       ... stepsize eta of gradient descent, default = " << default_config.eta << '\n'
              << "  -x, --early-exaggeration ... early exaggeration value, default = " << default_config.early_exaggeration << '\n'
              << "  -X, --late-exaggeration  ... late exaggeration value, default = " << default_config.late_exaggeration << '\n'
              << "  -i, --max-iter           ... number of gradient descent iterations, default = " << default_config.max_iter << '\n'
              << "  -s, --seed               ... random seed\n\n"

              << "  -k, --k-lo ... lower bound for k-means k, default = " << default_config.kmeans_lo << '\n'
              << "  -K, --k-hi ... upper bound for k-means k, default = " << default_config.kmeans_hi << "\n\n"

              << "  --num-hash-tables ... number of hash tables for FALCONN lsh. tunable parameter, default = " << default_config.num_hash_tables << '\n'
              << "  --num-hash-bits   ... number of hash bits, controls number of buckets per table. automatically set to max(16, log2(n)) if -1 is passed, default = " << default_config.num_hash_bits << '\n'
              << "  --num-probes      ... number of probes for multi-probe LSH. tunable parameter (inverse relation to L), default = " << default_config.num_probes << "\n\n"

              << "  --[no-]compute-objective  ... compute objective in every iteration, default = " << (default_config.compute_objective ? "on" : "off") << '\n'
              << "  --[no-]print_intermediate ... print intermediate embedding to file every iteration (for creating GIFs), default = " << (default_config.print_intermediate ? "on" : "off") << "\n\n"
              << "  --use-hyperplane          ... use hyperplane LSH (instead of cross-polytope), default = " << (default_config.use_hyperplane ? "on" : "off") << '\n'
              << "  --use-cross-polytope      ... use cross-polytope LSH (instead of hyperplane LSH), default = " << (!default_config.use_hyperplane ? "on" : "off") << '\n'

              << "  -h ... this message\n\n"

              << "and FILE is a csv file.\n\n"

              << "ktsne is an accelerated approximative version of tsne which uses LSH and kmeans in its computation.\n";
}

int main(int argc, char** argv) {
    config_t config;

    struct option long_opts[] = {
        {"compute-objective",     no_argument, &config.compute_objective, 1},
        {"no-compute-objective",  no_argument, &config.compute_objective, 0},
        {"print-intermediate",    no_argument, &config.print_intermediate, 1},
        {"no-print-intermediate", no_argument, &config.print_intermediate, 0},
        {"use-hyperplane-lsh",    no_argument, &config.use_hyperplane, 1},
        {"use-cross-polytope",    no_argument, &config.use_hyperplane, 0},
        {"random-init",           no_argument, &config.random_init, 1},
        {"pca-init",              no_argument, &config.random_init, 0},

        {"k-lo",                  required_argument, NULL, 'k'},
        {"k-hi",                  required_argument, NULL, 'K'},
        {"early-exaggeration",    required_argument, NULL, 'x'},
        {"late-exaggeration",     required_argument, NULL, 'X'},

        {"num-hash-tables",       required_argument, NULL, 1},
        {"num-hash-bits",         required_argument, NULL, 2},
        {"num-probes",            required_argument, NULL, 3},

        {"seed",                  required_argument, NULL, 's'},
        {"max-iter",              required_argument, NULL, 'i'},
        {NULL, 0, NULL, 0}
    };

    char c;
    while((c = getopt_long(argc, argv, "p:i:n:k:K:x:X:s:h", long_opts, NULL)) != -1) {
        switch(c) {
            case 0: /* flag */ break;
            case 1: config.num_hash_tables = std::atoi(optarg); break;
            case 2: config.num_hash_bits = std::atoi(optarg); break;
            case 3: config.num_probes = std::atoi(optarg); break;
            case 'p': config.perplexity = std::atoi(optarg); break;
            case 'n': config.eta = std::atof(optarg); break;
            case 'i': config.max_iter = std::atoll(optarg); break;
            case 'k': config.kmeans_lo = std::atoi(optarg); break;
            case 'K': config.kmeans_hi = std::atoi(optarg); break;
            case 'x': config.early_exaggeration = std::atof(optarg); break;
            case 'X': config.late_exaggeration = std::atof(optarg); break;
            case 's': config.seed = std::atoll(optarg); break;
            case 'h':
            default: print_usage(argv[0]); std::exit(-1);
        }
    }

    if(argc == optind) {
        std::cerr << "no file given!\n";
        print_usage(argv[0]);
        std::exit(-1);
    }

    std::mt19937 gen{ config.seed };
    std::vector<point_t> data = read_data(argv[optind]);
    std::vector<int> labels;

    normalize(data);
    center(data);

    size_t const n = data.size();
    size_t const d = 2;

    if(config.num_hash_bits == -1) { config.num_hash_bits = std::max(static_cast<size_t>(16), static_cast<size_t>(std::log2(n))); }

    std::clog << "configuration:\n"
              << "  perplexity: " << config.perplexity << '\n'
              << "  eta: " << config.eta << '\n'
              << "  early exaggeration: " << config.early_exaggeration << '\n'
              << "  late exaggeration: " << config.late_exaggeration << '\n'
              << "  iterations: " << config.max_iter << '\n'
              << "  kmeans k range: [" << config.kmeans_lo << ", " << config.kmeans_hi << "]\n"
              << "  initialization: " << (config.random_init ? "random" : "PCA") << "\n\n"

              << "  number of hash tables: " << config.num_hash_tables << '\n'
              << "  number of hash bits: " << config.num_hash_bits << '\n'
              << "  number of probes (multiprobing): " << config.num_probes << '\n'
              << "  LSH family: " << (config.use_hyperplane ? "hyperplane" : "cross-polytope") << "\n\n"

              << "  computing objective: " << (config.compute_objective ? "on" : "off") << '\n'
              << "  printing intermediate embeddings: " << (config.print_intermediate ? "on" : "off") << '\n'
              << "  seed: " << config.seed << "\n\n";

    Eigen::SparseMatrix<double> P_j_given_i = high_dimensional_affinities(data, config.perplexity, config.num_hash_tables, config.num_hash_bits, config.num_probes, config.use_hyperplane);
    Eigen::SparseMatrix<double> P_ij = Eigen::SparseMatrix<double>(P_j_given_i.transpose()) + P_j_given_i;
    P_ij /= P_ij.sum();

    Eigen::MatrixXdr Y(n, d);
    Eigen::MatrixXdr Y_(n, d);

    if(config.random_init) {
        initialize_gaussian(Y, 10e-4, gen);
    } else {
        initialize_PCA(Y, data);
    }

    Eigen::MatrixXdr iY    = Eigen::MatrixXdr::Zero(n, d);
    Eigen::MatrixXdr dY    = Eigen::MatrixXdr::Zero(n, d);
    Eigen::MatrixXdr gains = Eigen::MatrixXdr::Ones(n, d);

    std::uniform_int_distribution<size_t> k_dist{ config.kmeans_lo, config.kmeans_hi };

    if(config.compute_objective) { std::cout << "it,obj,normdY,procrustes\n"; }

    for(size_t it = 0; it < config.max_iter; ++it) {
        if(it % 100 == 0) { std::cerr << "it = " << it << '\n'; }
        Eigen::MatrixXdr F_attr = Eigen::MatrixXdr::Zero(n, d);

        double exaggeration = it < 0.25*config.max_iter ? 12 : 1; // artificially inflate P_ij value for first few iterations
        exaggeration = it > 0.9*config.max_iter ? 4 : 1;

        for(int k = 0; k < P_ij.outerSize(); ++k) {
            for(Eigen::SparseMatrix<double>::InnerIterator it{ P_ij, k }; it; ++it) {
                int i = it.row(), j = it.col();

                auto diff = Y.row(i) - Y.row(j);
                F_attr.row(i) += exaggeration*it.value() * (1/(1 + diff.squaredNorm())) * diff;
            }
        }

        // approximate F_rep by assigning cells using kmeans
        size_t k = k_dist(gen);
        auto [centroids, n_cell, point_assignments] = do_kmeans(Y, k, 10, gen, config.compute_objective);
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

        Y_ = Y;

        iY = momentum(it)*iY - config.eta*(gains.cwiseProduct(dY));
        Y += iY;

        Eigen::RowVectorXd Y_mean = Y.colwise().mean();
        Y = Y.rowwise() - Y_mean;

        if(config.compute_objective) { // warning: this is slow!
            double kl = 0;

            for(int k = 0; k < P_ij.outerSize(); ++k) {
                for(Eigen::SparseMatrix<double>::InnerIterator it{ P_ij, k }; it; ++it) {
                    int i = it.row(), j = it.col();
                    double q_icell = std::sqrt(q_icellZ_sq(i, point_assignments[j]) / Z_est);

                    kl += it.value()*std::log2(it.value() / q_icell);
                }
            }

            double procrustes_error = compute_procrustes(Y_, Y);

            std::cout << it << "," << kl << "," << dY.norm() << "," << procrustes_error << '\n';
        }

        if(config.print_intermediate && it % 5 == 0) {
            fs::create_directory("gif");
            std::string outname = std::string{ "gif/" } + fs::path(argv[optind]).stem().string() + "_embedding_it_" + std::to_string(it) + ".csv";
            print_csv(outname, Y, labels);
        }
    }

    std::stringstream outname_ss;
    outname_ss << "ktsne_" << fs::path(argv[optind]).stem().string() << "_embedding_eta_" << config.eta << "_p_" << config.perplexity << "_s_" << config.seed << ".csv";

    print_csv(outname_ss.str(), Y, labels);
}
