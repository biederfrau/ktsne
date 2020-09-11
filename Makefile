ktsne: ktsne.cpp
	g++ ktsne.cpp -Ifalconn/external/eigen -Ifalconn/external/simple-serializer -Ifalconn/src/include -march=native -o ktsne -pthread -g -std=c++17

test: ktsne.cpp
	g++ ktsne.cpp -Ifalconn/external/eigen -Ifalconn/external/simple-serializer -Ifalconn/src/include -march=native -o ktsne -pthread -g -std=c++17
	./ktsne -v iris_no_labels.csv
