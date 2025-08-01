#include<bits/stdc++.h>
#define int long long
using namespace std;
double truck_speed, drone_speed;// toc do xe tai va drone
int capacity; // suc chua cua xe tai
int num_nodes, total_demand, num_clusters; // so luong node, tong so yeu cau = tong so khach hang (1 khach hang yeu cau 1 goi hang), so luong cluster
struct Node 
{
	double x;
	double y;
	string name;
} node[1010], centroid[1010]; 
// centroid: toa do cua tam cac cluster
// node: toa do cac node. node[0] la diem bat dau
int getRandomNumber(int n) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<> distr(1, n);
    return distr(gen);
}

double getRandomUniformNumber(double n) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_real_distribution<double> distr(0.0, n);
    return distr(gen);
}
int cluster_assignment[1010]; // index cluster cua node
vector<int> cluster[1010]; // danh sach cac node cua cac cluster
int cluster_size[1010]; // size cua cac cluster
void K_means() 
{
    auto start_time = chrono::high_resolution_clock::now();
    auto timeout = chrono::seconds(5); // 5 sec timeout
    
    // Best solution tracking
    double best_sum_squared_distance = DBL_MAX;
    int best_cluster_assignment[1010];
    Node best_centroids[1010];
    vector<int> best_clusters[1010];
    int best_cluster_size[1010];
    
    int run_count = 0;
    
    while (true) {
        // Check if 5 seconds has elapsed
        auto current_time = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::milliseconds>(current_time - start_time) >= timeout) {
            break;
        }
        
        run_count++;
        
        // Reset for current run
        for(int i = 1; i <= num_nodes; i++) {
            cluster_assignment[i] = -1;
        }
        for(int i = 0; i < num_clusters; i++) {
            cluster_size[i] = 0;
            cluster[i].clear();
        }
        
        int changed = 1;
        int iterations = 0;
        
        // Random initialization of centroids
        for (int i = 0; i < num_clusters; ++i) {
            int random_index = getRandomNumber(num_nodes - 1) + 1; 
            centroid[i] = node[random_index];
        }
        
        // K-means clustering iterations
        while (changed && iterations < 100) {
            changed = 0;
            iterations++;
            
            for (int i = 2; i <= num_nodes; ++i) { 
                double min_distance = DBL_MAX;
                int nearest_centroid = -1;
                
                for (int j = 0; j < num_clusters; ++j) {
                    double distance = sqrt((node[i].x - centroid[j].x) * (node[i].x - centroid[j].x) + 
                                         (node[i].y - centroid[j].y) * (node[i].y - centroid[j].y));
                    if (distance < min_distance) {
                        min_distance = distance;
                        nearest_centroid = j;
                    }
                }
                
                if (cluster_assignment[i] != nearest_centroid) {
                    changed = 1;
                    cluster_assignment[i] = nearest_centroid;
                }
            }
            
            // Update centroids
            vector<Node> new_centroid(num_clusters, {0.0, 0.0, ""});
            for(int i = 0; i < num_clusters; i++) {
                cluster_size[i] = 0;
            }
            
            for (int i = 2; i <= num_nodes; ++i) {
                int cluster = cluster_assignment[i];
                new_centroid[cluster].x += node[i].x;
                new_centroid[cluster].y += node[i].y;
                cluster_size[cluster]++;
            }
            
            for (int j = 0; j < num_clusters; j++) {
                if (cluster_size[j] > 0) {
                    new_centroid[j].x /= cluster_size[j];
                    new_centroid[j].y /= cluster_size[j];
                }
            }
            
            for (int j = 0; j < num_clusters; j++) {
                centroid[j] = new_centroid[j];
            }
        }
        
        // Capacity adjustment phase
        int adjustments_needed = 1;
        while (adjustments_needed) {
            adjustments_needed = 0;
            
            for (int j = 0; j < num_clusters; ++j) {
                while (cluster_size[j] > capacity) {
                    adjustments_needed = 1;
                    double max_distance = -1;
                    int farthest_node = -1;
                    
                    for (int i = 2; i <= num_nodes; ++i) {
                        if (cluster_assignment[i] == j) {
                            double distance = sqrt((node[i].x - centroid[j].x) * (node[i].x - centroid[j].x) + 
                                                 (node[i].y - centroid[j].y) * (node[i].y - centroid[j].y));
                            if (distance > max_distance) {
                                max_distance = distance;
                                farthest_node = i;
                            }
                        }
                    }
                    
                    double min_distance = DBL_MAX;
                    int new_cluster = -1;
                    for (int k = 0; k < num_clusters; ++k) {
                        if (k != j) { 
                            double distance = sqrt((node[farthest_node].x - centroid[k].x) * (node[farthest_node].x - centroid[k].x) + 
                                                 (node[farthest_node].y - centroid[k].y) * (node[farthest_node].y - centroid[k].y));
                            if (distance < min_distance) {
                                min_distance = distance;
                                new_cluster = k;
                            }
                        }
                    }
                    
                    cluster_assignment[farthest_node] = new_cluster;
                    cluster_size[j]--;
                    cluster_size[new_cluster]++;
                }
            }
            
            // Recalculate centroids after capacity adjustments
            vector<Node> new_centroid(num_clusters, {0.0, 0.0, ""});
            for(int i = 0; i < num_clusters; i++) {
                cluster_size[i] = 0;
            }
            
            for (int i = 2; i <= num_nodes; ++i) {
                int cluster = cluster_assignment[i];
                new_centroid[cluster].x += node[i].x;
                new_centroid[cluster].y += node[i].y;
                cluster_size[cluster]++;
            }

            for (int j = 0; j < num_clusters; ++j) {
                if (cluster_size[j] > 0) {
                    new_centroid[j].x /= cluster_size[j];
                    new_centroid[j].y /= cluster_size[j];
                }
            }

            for (int j = 0; j < num_clusters; ++j) {
                centroid[j] = new_centroid[j];
            }
        }
        
        // Calculate sum of squared distances for current solution
        double current_sum_squared_distance = 0.0;
        for (int i = 2; i <= num_nodes; i++) {
            int cluster_id = cluster_assignment[i];
            double dx = node[i].x - centroid[cluster_id].x;
            double dy = node[i].y - centroid[cluster_id].y;
            current_sum_squared_distance += (dx * dx + dy * dy);
        }
        
        // Update best solution if current is better
        if (current_sum_squared_distance < best_sum_squared_distance) {
            best_sum_squared_distance = current_sum_squared_distance;
            
            // Save best cluster assignments
            for(int i = 1; i <= num_nodes; i++) {
                best_cluster_assignment[i] = cluster_assignment[i];
            }
            
            // Save best centroids
            for(int i = 0; i < num_clusters; i++) {
                best_centroids[i] = centroid[i];
                best_cluster_size[i] = cluster_size[i];
            }
            
            // Save best clusters
            for(int i = 0; i < num_clusters; i++) {
                best_clusters[i].clear();
            }
            for (int i = 2; i <= num_nodes; i++) {
                best_clusters[cluster_assignment[i]].push_back(i);
            }
        }
        
        // Clear clusters for next iteration
        for(int i = 0; i < num_clusters; i++) {
            cluster[i].clear();
        }
    }
    
    // Restore best solution to global variables
    for(int i = 1; i <= num_nodes; i++) {
        cluster_assignment[i] = best_cluster_assignment[i];
    }
    
    for(int i = 0; i < num_clusters; i++) {
        centroid[i] = best_centroids[i];
        cluster_size[i] = best_cluster_size[i];
        cluster[i] = best_clusters[i];
    }
    
    //cout << "K-means completed after " << run_count << " runs in 5 seconds." << endl;
    //cout << "Best sum of squared distances: " << best_sum_squared_distance << endl;
}
double p[2][1010][1010];
double d[1010][1010];
int flag[1010];
int num_ants;
double beta = 2, alpha = 1;
double t0;
double p1 = 1, p2 = 0, r0 = 0;
double global_best_fitness = 1e18;
vector<int>global_best_route;
int roulette_wheel_selection(int u, const std::set<int>& s, int type) 
{
    std::map<int, double> probabilities;
    double sum = 0.0;

    for (int v : s) {
        if (d[u][v] == 0) continue;
        probabilities[v] = pow(p[type][u][v], alpha) / pow(d[u][v], beta);
        sum += probabilities[v];
    }
    
    if (sum == 0.0) return -1; 

    for (auto& elem : probabilities) {
        elem.second /= sum;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + std::random_device()();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0, 1);
    double random_number = dis(gen);
    double cumulative_probability = 0.0;

    for (auto& elem : probabilities) 
	{
        cumulative_probability += elem.second;
        if (random_number <= cumulative_probability)
		{
            return elem.first;
        }
    }

    return -1; 
}
int absolute_selection(int u, const std::set<int>& s, int type) 
{
    std::map<int, double> probabilities;
    int chosen = 0;
    double max_prob = -1;
    for (int v : s) 
	{
        if (d[u][v] == 0) continue; 
        probabilities[v] = pow(p[type][u][v], alpha) / pow(d[u][v], beta);
        if(probabilities[v] > max_prob)
        {
        	max_prob = probabilities[v];
        	chosen = v;
		}
    }
    return chosen;
}
double get_distance(int x, int y)
{
	int xd = node[x].x - node[y].x;
	int yd = node[x].y - node[y].y;
	return sqrt(xd * xd + yd * yd);
}
void update_local_pheno(int u, int v, int type)
{
	p[type][u][v] = (1 - p2) * p[type][u][v] + p2 * t0;
}
void update_global_pheno()
{
	for(int i = 1; i <= num_nodes; i++)
	{
		for(int j = 1; j <= num_nodes; j++)
		{
			p[0][i][j] = (1 - p1) * p[0][i][j];
			p[1][i][j] = (1 - p1) * p[1][i][j];
		}
	}
	if(global_best_route.size() >= 1)
	{
		p[0][1][global_best_route[0]] += p1 * 1.0 / global_best_fitness;
	}
	for(int i = 0; i < global_best_route.size() - 2; i+=2)
	{
		p[0][global_best_route[i]][global_best_route[i + 2]] += p1 * 1.0 / global_best_fitness;
	}
	for(int i = 0; i < global_best_route.size() - 1; i+=2)
	{
		p[1][global_best_route[i]][global_best_route[i + 1]] += p1 * 1.0 / global_best_fitness;
	}
}
double fitness_eval(vector<int>v)
{
	double fitt = 0;
	fitt += d[1][v[0]] / truck_speed;
	for(int i = 0; i < v.size() - 2; i+=2)
	{
		fitt += max(d[v[i]][v[i + 2]] / truck_speed, (d[v[i]][v[i + 1]] + d[v[i + 1]][v[i + 2]]) / drone_speed);
	}
	if(v.size() % 2 == 0)
	{
		int i = v.size() - 2;
		fitt += max(d[v[i]][1] / truck_speed, (d[v[i]][v[i + 1]] + d[v[i + 1]][1]) / drone_speed);
	}
	else
	{
		int i = v.size() - 1;
		fitt += d[v[i]][1] / truck_speed;
	}
	return fitt;
}
void route_construction(vector<int>v)
{
    // Add timing for ACS algorithm
    auto acs_start_time = chrono::high_resolution_clock::now();
    auto acs_timeout = chrono::seconds(5); // 5 seconds timeout for ACS
    
	global_best_fitness = 1e18;
	global_best_route.clear();
	num_ants = v.size() + 1;
	int cur = 1;
	double l = 0;
	for(int i = 1; i <= num_nodes; i++)
	{
		flag[i] = 0;
	}
	while(1)
	{
		//cout << cur << " ";
		double min_dis = 1e18;
		int new_cur = 1;
		for(int u:v)
		{
			if(flag[u] == 1) continue;
			if(d[cur][u] < min_dis)
			{
				min_dis = d[cur][u];
				new_cur = u;
			}
		}
		l += d[cur][new_cur];
		flag[new_cur] = 1;
		cur = new_cur;
		if(cur == 1) break;
	}
	//cout << "\n";
	t0 = num_ants * 1.0 / l;
	//cout << num_ants << " " << l << " " << t0 << '\n';
	p1 = 1; p2 = 0; r0 = 0;
	for(int i = 1; i <= num_nodes; i++)
	{
		for(int j = 1; j <= num_nodes; j++)
		{
			p[0][i][j] = getRandomUniformNumber(t0);
			p[1][i][j] = getRandomUniformNumber(t0);
		}
	}
	
	// Replace fixed iteration loop with time-based loop
	int iteration = 0;
	while (true) {
        // Check if 5 seconds have elapsed
        auto current_time = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::milliseconds>(current_time - acs_start_time) >= acs_timeout) {
            break;
        }
        
        iteration++;
        
        // Calculate progress based on time (0.0 to 1.0)
        double time_elapsed = chrono::duration_cast<chrono::milliseconds>(current_time - acs_start_time).count();
        double progress = min(1.0, time_elapsed / 5000.0); // 5000ms = 5 seconds
        
        // Update parameters based on time progress instead of iteration
        r0 = progress;
        p1 = 1.0 - progress;
        p2 = progress;
        
		vector<vector<int> > route;
		for(int ant = 1; ant <= num_ants; ant+=2)
		{
			set<int>s;
			for(int u: v)
			{
				s.insert(u);
			}
			vector<int> r;
			int ant1 = 1, ant2 = 1;
			while(s.size())
			{
				int chosen;
				if(getRandomUniformNumber(1.0) <= r0)
				{
					chosen = absolute_selection(ant1, s, 0);
				}
				else
				{
					chosen = roulette_wheel_selection(ant1, s, 0);
				}
				r.push_back(chosen);
				s.erase(chosen);
				update_local_pheno(ant1, chosen, 0);
				ant1 = chosen;
				if(s.size() == 0) break;
				ant2 = ant1;
				if(getRandomUniformNumber(1.0) <= r0)
				{
					chosen = absolute_selection(ant2, s, 1);
				}
				else
				{
					chosen = roulette_wheel_selection(ant2, s, 1);
				}
				r.push_back(chosen);
				s.erase(chosen);
				update_local_pheno(ant2, chosen, 1);
				ant2 = chosen;
			}
			route.push_back(r);
		}
		double local_best_fitness = 1e18;
		vector<int> local_best_route;
		for(auto r:route)
		{
			double ant_fitness = fitness_eval(r);
			if(ant_fitness < local_best_fitness)
			{
				local_best_fitness = ant_fitness;
				local_best_route.clear();
				for(int i:r)
				{
					local_best_route.push_back(i);
				}
			}
		}
		if(local_best_fitness < global_best_fitness)
		{
			global_best_fitness = local_best_fitness;
			global_best_route.clear();
			for(int i:local_best_route)
			{
				global_best_route.push_back(i);
			}
		}
		update_global_pheno();
	}
	
	//cout << "ACS completed after " << iteration << " iterations in 5 seconds." << endl;
}
int32_t main()
{
	freopen("uniform-5-n500.txt", "r", stdin);
	//freopen("uniform-1-n250.out", "w", stdout);
	string s;
	getline(cin, s);
	cin >> truck_speed;
	cin.ignore(); 
	getline(cin, s);
	cin >> drone_speed;
	cin.ignore(); 
	getline(cin, s);
	cin >> num_nodes;
	cin.ignore(); 
	getline(cin, s);
	cin >> node[1].x >> node[1].y >> node[1].name;
	cin.ignore(); 
	getline(cin, s);
	for(int i = 2; i <= num_nodes; i++)
	{
		cin >> node[i].x >> node[i].y >> node[i].name;
	}
	if(num_nodes <= 100) capacity = 40;
	else capacity = 100;
	total_demand = num_nodes - 1;
	num_clusters = (total_demand - 1) / capacity + 1;
	truck_speed = 13; drone_speed = 20;
//	cout << truck_speed << " " << drone_speed << " " << num_nodes << '\n';
//	for(int i = 1; i <= num_nodes; i++)
//	{
//		cout << node[i].x << " " << node[i].y << " " << node[i].name << '\n';
//	}
	K_means();
//	for(int i = 0; i < num_clusters; i++)
//	{
//		cout << "cluster " << i <<'\n';
//		for(int j:cluster[i])
//		{
//			cout << j << " ";
//		}
//		cout << '\n';
//	}
	for(int i = 1; i <= num_nodes; i++)
	{
		for(int j = 1; j <= num_nodes; j++)
		{
			d[i][j] = get_distance(i, j);
			//cout << i << " " << j << " " << d[i][j] << '\n';
		}
	}
	double ans = -1;
	for(int i = 0; i < num_clusters; i++)
	{
		route_construction(cluster[i]);
		for(int u:global_best_route)
		{
			//cout << u << " " << node[u].x << " " << node[u].y << '\n';
		}
		ans = max(ans, global_best_fitness);
	}
	cout << ans;
}
