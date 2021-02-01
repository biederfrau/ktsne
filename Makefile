OPT_FLAGS := -O4 -floop-parallelize-all -floop-nest-optimize -floop-flatten -floop-interchange -floop-block -flto -ffast-math -funroll-loops -march=native -mtune=native -faggressive-loop-optimizations -pthread
EXT := -Ifalconn/external/eigen -Ifalconn/external/simple-serializer -Ifalconn/src/include

ktsne: ktsne.cpp
	g++ ktsne.cpp $(EXT) -std=c++17 -lstdc++fs -DNDEBUG $(OPT_FLAGS) -o ktsne
