g++ -fopenmp -lgomp -std=c++14 -O3 -o monte_carlo_random monte_carlo_IC_random.cpp
g++ -fopenmp -lgomp -std=c++14 -O3 -o monte_carlo_degree monte_carlo_IC_degree.cpp
g++ -fopenmp -lgomp -std=c++14 -O3 -o monte_carlo_random monte_carlo_LT_random.cpp
g++ -fopenmp -lgomp -std=c++14 -O3 -o monte_carlo_degree monte_carlo_LT_degree.cpp
g++ -fopenmp -lgomp -std=c++14 -O3 -o test test.cpp