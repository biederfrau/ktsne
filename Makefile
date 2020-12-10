ktsne: ktsne.cpp
	g++ ktsne.cpp -Ifalconn/external/eigen -Ifalconn/external/simple-serializer -Ifalconn/src/include -march=native -o ktsne -pthread -O3 -std=c++17 -lstdc++fs
