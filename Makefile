OPT_FLAGS := -O4 -floop-parallelize-all -floop-nest-optimize -floop-flatten -floop-interchange -floop-block -flto -ffast-math -funroll-loops -march=native -mtune=native -faggressive-loop-optimizations
EXT := -Ifalconn/external/eigen -Ifalconn/external/simple-serializer -Ifalconn/src/include

ktsne: ktsne.cpp
	g++ ktsne.cpp $(EXT) -std=c++17 -lstdc++fs -DNDEBUG $(OPT_FLAGS) -pthread -o ktsne -DEIGEN_DONT_PARALLELIZE

ktsne_debug: ktsne.cpp
	g++ ktsne.cpp $(EXT) -std=c++17 -lstdc++fs -DNDEBUG -Og -pthread -o ktsne_debug -DEIGEN_DONT_PARALLELIZE -g

binom_bench:
	g++ benchmark_eigen_pairwise_dist.cpp -Ifalconn/external/eigen -o binom_bench -std=c++17 -DNDEBUG $(OPT_FLAGS) -DEIGEN_DONT_PARALLELIZE
