#include<bits/stdc++.h>
#define int long long
using namespace std;
double truck_speed, drone_speed;// toc do xe tai va drone
int capacity; // suc chua cua xe tai
int num_nodes, total_demand, num_clusters; // so luong node, tong so yeu cau = tong so khach hang (1 khach hang yeu cau 1 goi hang), so luong cluster
double ans = 1e18;
double d[1010][1010];
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
double get_distance(int x, int y)
{
	int xd = node[x].x - node[y].x;
	int yd = node[x].y - node[y].y;
	return sqrt(xd * xd + yd * yd);
}
double fitness_eval(vector<int>v)
{
	for(int i:v)
	{
		cout << i << " ";
	}
	cout << '\n';
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
int bt[100], ok[100];
void f(int next_node)
{
	if(next_node > num_nodes) 
	{
		vector<int>v;
		for(int i = 2; i <= num_nodes; i++)
		{
			v.push_back(bt[i]);
		}
		ans = min(ans, fitness_eval(v));
		return;
	}
	for(int i = 2; i <= num_nodes; i++)
	{
		if(ok[i] == 1) continue;
		bt[next_node] = i;
		ok[i] = 1;
		f(next_node + 1);
		ok[i] = 0;
		bt[next_node] = 0;
	}
}
int32_t main()
{
	freopen("uniform-1-n5.txt", "r", stdin);
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
	//K_means();
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
	bt[1] = 1;
	ok[1] = 1;
	f(2);
	cout << ans;
}
