// essential
#include <stdio.h>
#include <omp.h>

// random
#include <random>

// data structure
#include <vector>
#include <queue>
#include <set>
#include <bitset>

#include <string.h>
#include <string>

#include <algorithm>
#include <chrono>

#include <iostream>

#define BATCH_SIZE 20
#define MAX_N 20000
#define TEST_COUNT 100000
#define MIN_SEED_SIZE 1
#define MAX_ITER 100

int n, m;
std::vector<int> V[BATCH_SIZE];
std::vector< std::pair<int, double> > E[BATCH_SIZE][MAX_N]; // adjacency list
std::bitset<MAX_N> pi[BATCH_SIZE][MAX_ITER+1];
int tot[MAX_ITER+1][MAX_N];

std::queue<int> que[BATCH_SIZE];

// Threshold
std::vector<double> threshold[BATCH_SIZE];
std::vector<double> act[BATCH_SIZE][MAX_ITER+1];

std::random_device rd[MAX_N];
std::mt19937 gen[MAX_N];
std::uniform_real_distribution<double> unif[MAX_N];


int limit_iter;

std::set<int> pset;
long long mc_result[BATCH_SIZE];

// MC simulation function
int do_task(int i, int real_seed_size){
	pi[i][0].reset();

	// set initial seeds
    for(int j=0;j<real_seed_size;j++){
		pi[i][0].flip(V[i][j]);
		que[i].push(V[i][j]);
	}

	/*LT*/
	// set thersholds for generating random numbers 
	for(int j=0;j<std::min(n, MAX_N);j++){
		threshold[i][j] = (unif[i])(gen[j]);
		act[i][0][j] = (double)0.0;
	}

	int ret = 0;
	int iter = 1;
	while(!que[i].empty() && iter <= limit_iter){
		pi[i][iter] = pi[i][iter-1];
		int lft = (int)que[i].size();
		ret += lft;
		
		act[i][iter].resize(n);
		act[i][iter] = act[i][iter-1];

		while(lft--){
			int now = que[i].front(); que[i].pop();
			for(auto &nxt: E[i][now]){ 
				if(pi[i][iter].test(nxt.first))	continue;
				act[i][iter][nxt.first] += nxt.second;
				//LT cascading behavior
				if(act[i][iter][nxt.first] > threshold[i][nxt.first]){
					pi[i][iter].flip(nxt.first);
                    que[i].push(nxt.first);
                }
			}
		}
		iter++;
	}
    // When MC simulation is not finished...
	if(!que[i].empty()){
		while(!que[i].empty()) que[i].pop();
		return -10000000;
	}
	for(int j=iter;j<=limit_iter;j++){
		pi[i][j] = pi[i][iter-1];
	}
	return ret;
}


int main(int argc, char **argv){
	omp_set_num_threads(4);
    // initialization for generating random numbers
	for(int i=0;i<MAX_N;i++){
		gen[i] = std::mt19937((rd[i])());
		unif[i] = std::uniform_real_distribution<>(0.0, 1.0);
	}


	FILE *in = fopen(argv[1], "r"); 
	if(!in){
		printf("FILE NOT EXIST!\n");
		return 0;
	}
	fscanf(in, "%d%d", &n, &m); // n: |V|, m: |E|
	
	printf("%d %d\n", n, m);
	V[0].resize(n);
	/*LT*/
	threshold[0].resize(n);
	for(int i=0;i<BATCH_SIZE;i++){
		act[i][0].resize(n);
	}
	/*LT*/
	for(int i=0;i<n;i++){
		V[0][i] = i;
		/*LT*/
		threshold[0][i] = (double)0.0;
		act[0][0][i] = (double)0.0;
		/*LT*/
	}

    // generate adjacency list
	for(int i=0;i<m;i++){
		int from, to; double prob;
		fscanf(in, "%d%d%lf", &from, &to, &prob);
		for(int j=0;j<BATCH_SIZE;j++){
			E[j][from].push_back({to, prob});
		}
	}
	fclose(in);

	int len_path = strlen(argv[1]);
	std::string input_path(argv[1]);
	char output_path[100] = "raw_data/";
	input_path.copy(output_path+9, len_path-11, 7);
	printf("output_path: %s\n", output_path);
	
    limit_iter = std::min(n, MAX_ITER);

	int turn = 0;
    while(turn < 2000){
		printf("turn : %d\n",turn);
		++turn;
        // select seeds randomly
		auto it = V[0].begin();
        std::shuffle(it, V[0].end(), gen[0]);
		for(int i=1;i<BATCH_SIZE;i++){
			V[i] = V[0];
			threshold[i] = threshold[0];
			act[i][0]=act[0][0];
		}
		for(int i=0;i<=limit_iter;i++){
			for(int j=0;j<n;j++){
				tot[i][j] = 0;
			}
		}
		for(int i=0;i<BATCH_SIZE;i++){
			mc_result[i] = 0;
		}
        // determine size of seedset (<= 2% of total nodes)
		int seed_size = (((unif[0])(gen[0])) * (0.02 * n - MIN_SEED_SIZE)) + MIN_SEED_SIZE;
		for(int tc=0;tc<TEST_COUNT;tc+=BATCH_SIZE){
			int bsize = std::min(BATCH_SIZE, (TEST_COUNT-tc));
			#pragma omp parallel for
			
			for(int i=0;i<bsize;i++){
                // run MC simulations
				int ret = do_task(i, seed_size);
				if(ret < 0 || mc_result[i] < 0) mc_result[i] = -10000000;
				else mc_result[i] += ret;
			}
			
			for(int i=0;i<bsize;i++){
				if(mc_result[i] < 0){
					printf("ERROR!\n");
					return 0;
				}
				for(int ii=0;ii<=limit_iter;ii++){
					#pragma omp parallel for
					for(int jj=0;jj<n;jj++){
						tot[ii][jj] += pi[i][ii].test(jj);
					}
				}
			}
		}
		
		// output simluation results
        long long sum = 0;
		for(int i=0;i<BATCH_SIZE;i++) sum += mc_result[i];
		sprintf(output_path + (len_path - 2), "/%d.txt", turn); /*LT*/
		FILE *out = fopen(output_path, "w");
		for(int i=0;i<=limit_iter;i++){
			for(int j=0;j<n;j++){
				fprintf(out, "%.4f ", tot[i][j] / (double)TEST_COUNT);
			}
			fprintf(out, "\n");
			fflush(out);
		}
		fclose(out);
	}
	return 0;
}