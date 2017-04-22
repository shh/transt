#pragma once
#include <boost/progress.hpp>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <iterator>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <armadillo>
#include <map>
#include <set>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <bitset>
#include <queue>
#include <boost/function.hpp>
#include <iterator>
#include <omp.h>
using namespace std;
using namespace arma;


class Dataset
{
public:
	const string	base_dir;
	const string	training;
	const string	developing;
	const string	testing;
	const string	type;
	const string	name;
	const bool	self_false_sampling;
public:
	Dataset(const string& name,
		const string& base_dir,
		const string& training,
		const string& developing,
		const string& testing,
		const string& type,
		const bool& self_false_sampling)
		:name(name),
		base_dir(base_dir),
		training(training),
		developing(developing),
		testing(testing),
		type(type),
		self_false_sampling(self_false_sampling)
	{
		;
	}
};
enum TaskType
{
	LinkPredictionHead,
	LinkPredictionTail,
	LinkPredictionRelation,
	TripletClassification
};
template<typename T> class storage_vmat
{
public:
	static void save(const vector<Mat<T>>& vmatout, ofstream& fout)
	{
		auto n_size = vmatout.size();
		fout.write((char*)&n_size, sizeof(vmatout.size()));
		for (const Mat<T> & ivmatout : vmatout)
		{
			fout.write((char*)&ivmatout.n_rows, sizeof(ivmatout.n_rows));
			fout.write((char*)&ivmatout.n_cols, sizeof(ivmatout.n_cols));
			fout.write((char*)ivmatout.memptr(), ivmatout.n_elem * sizeof(T));
		}
	}
	static void load(vector<Mat<T>>& vmatin, ifstream& fin)
	{
		arma::uword n_size;
		fin.read((char*)&n_size, sizeof(n_size));
		vmatin.resize(n_size);
		for (Mat<T> & ivmatin : vmatin)
		{
			arma::uword	n_row, n_col;
			fin.read((char*)&n_row, sizeof(n_row));
			fin.read((char*)&n_col, sizeof(n_col));
			ivmatin.resize(n_row, n_col);
			fin.read((char*)ivmatin.memptr(), n_row * n_col * sizeof(T));
		}
	}
	static void save(const vector<Col<T>>& vmatout, ofstream& fout)
	{
		auto n_size = vmatout.size();
		fout.write((char*)&n_size, sizeof(vmatout.size()));

		for (const Col<T> & ivmatout : vmatout)
		{
			fout.write((char*)&ivmatout.n_rows, sizeof(ivmatout.n_rows));
			fout.write((char*)&ivmatout.n_cols, sizeof(ivmatout.n_cols));
			fout.write((char*)ivmatout.memptr(), ivmatout.n_elem * sizeof(T));
		}
	}
	static void load(vector<Col<T>>& vmatin, ifstream& fin)
	{
		arma::uword n_size;
		fin.read((char*)&n_size, sizeof(n_size));
		//cout << "n_size=" << n_size << endl;
		vmatin.resize(n_size);
		for (Col<T> & ivmatin : vmatin)
		{
			arma::uword	n_row, n_col;
			fin.read((char*)&n_row, sizeof(n_row));
			fin.read((char*)&n_col, sizeof(n_col));
			//cout << n_row << "," << n_col << endl;
			ivmatin.resize(n_row);
			fin.read((char*)ivmatin.memptr(), n_row * n_col * sizeof(T));
			//ivmatin.print();
		}
	}
};
template<typename T> class storage_vec
{
public:
	static void save(const Col<T>& vscr, ofstream& fout)
	{
		fout.write((char*)&vscr.n_rows, sizeof(vscr.n_rows));
		fout.write((char*)&vscr.n_cols, sizeof(vscr.n_cols));
		fout.write((char*)vscr.memptr(), vscr.n_elem * sizeof(T));
	}

	static void load(Col<T>& vscr, ifstream& fout)
	{
		arma::uword	n_row, n_col;
		fout.read((char*)&n_row, sizeof(n_row));
		fout.read((char*)&n_col, sizeof(n_col));
		vscr.resize(n_row);
		fout.read((char*)vscr.memptr(), n_row * n_col * sizeof(T));
	}
};
template<typename T> class storage_vector
{
public:
	static void save(const vector<T>& vscr, ofstream& fout)
	{
		//auto n_size = vmatout.size();
		auto n_size = vscr.size();
		//fout.write((char*)&n_size, sizeof(vmatout.size()));
		fout.write((char*)&n_size, sizeof(vscr.size()));
		for (auto i = vscr.begin(); i != vscr.end(); ++i)
		{
			fout.write((char*)&(*i), sizeof(T));
		}
	}
	static void load(vector<T>& vscr, ifstream& fout)
	{
		arma::uword	n_size;
		//fout.read((char*)&n_size, sizeof(vmatout.size()));
		fout.read((char*)&n_size, sizeof(vscr.size()));
		vscr.resize(n_size);
		for (auto i = vscr.begin(); i != vscr.end(); ++i)
		{
			//fout.write((char*)&(*i), sizeof(T));
			fout.read((char*)&(*i), sizeof(T));
		}
	}
};
class DataModel
{
public:
	Dataset dataset;
public:
	set<pair<pair<int, int>, int> >		check_data_train;
	set<pair<pair<int, int>, int> >		check_data_all;
public:
	vector<pair<pair<int, int>, int> >	data_train;
	vector<pair<pair<int, int>, int> >	data_dev_true;
	vector<pair<pair<int, int>, int> >	data_dev_false;
	vector<pair<pair<int, int>, int> >	data_test_true;
	vector<pair<pair<int, int>, int> >	data_test_false;
public:
	set<int>			set_tail;
	set<int>			set_head;
	set<string>			set_entity;
	set<string>			set_relation;
public:
	vector<set<int>>	set_relation_tail;
	vector<set<int>>	set_relation_head;
public:
	vector<int>	relation_type;
public:
	vector<string>		entity_id_to_name;
	vector<string>		relation_id_to_name;
	vector<string>		type_id_to_name;
	map<string, int>	entity_name_to_id;
	map<string, int>	relation_name_to_id;
	map<string, int>	type_name_to_id;
	map<int, set<int> >	type_of_entity;
	map<int, vector<int> >  type2entity; //type:entity1;entity2;entity3
	map<int, pair<set<int>, set<int> > > type_of_relation;
	map<int, pair<int, int > > single_type_of_relation;
	vector<vector<vector<double> > > rel_ent_pos_typescore;
	vector<double>		prob_head;
	vector<double>		prob_tail;
	vector<double>		relation_tph;
	vector<double>		relation_hpt;
	map<string, int>	count_entity;
	map<pair<int, int>, int> count_hr;
	map<pair<int, int>, int> count_rt;
	map<pair<int, int>, int> count_ht;
	map<int, map<int, int> >	tails;
	map<int, map<int, int> >	heads;
	map<int, map<int, vector<int> > >     rel_heads;
	map<int, map<int, vector<int> > >     rel_tails;
	map<pair<int, int>, int>		     rel_finder;
public:
	int zeroshot_pointer;
public:
	DataModel(const Dataset& dataset) :dataset(dataset)
	{
		load_training(dataset.base_dir + dataset.training);
		relation_hpt.resize(set_relation.size());
		relation_tph.resize(set_relation.size());
		for (auto i = 0; i != set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for (auto ds = rel_heads[i].begin(); ds != rel_heads[i].end(); ++ds)
			{
				++sum;
				total += ds->second.size();
			}
			relation_tph[i] = total / sum;
		}
		for (auto i = 0; i != set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for (auto ds = rel_tails[i].begin(); ds != rel_tails[i].end(); ++ds)
			{
				++sum;
				total += ds->second.size();
			}
			relation_hpt[i] = total / sum;
		}
		zeroshot_pointer = set_entity.size();
		load_testing(dataset.base_dir + dataset.developing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(dataset.base_dir + dataset.testing, data_test_true, data_test_false, dataset.self_false_sampling);
		set_relation_head.resize(set_entity.size());
		set_relation_tail.resize(set_relation.size());
		prob_head.resize(set_entity.size());
		prob_tail.resize(set_entity.size());
		for (auto i = data_train.begin(); i != data_train.end(); ++i)
		{
			++prob_head[i->first.first];
			++prob_tail[i->first.second];

			++tails[i->second][i->first.first];
			++heads[i->second][i->first.second];

			set_relation_head[i->second].insert(i->first.first);
			set_relation_tail[i->second].insert(i->first.second);
		}
#pragma omp parallel for
#pragma ivdep
		for (auto elem = prob_head.begin(); elem != prob_head.end(); ++elem)
		{
			*elem /= data_train.size();
		}
#pragma omp parallel for
#pragma ivdep
		for (auto elem = prob_tail.begin(); elem != prob_tail.end(); ++elem)
		{
			*elem /= data_train.size();
		}
		double threshold = 1.5;
		relation_type.resize(set_relation.size());
		for (auto i = 0; i<set_relation.size(); ++i)
		{
			if (relation_tph[i]<threshold && relation_hpt[i]<threshold)
			{
				relation_type[i] = 1;
			}
			else if (relation_hpt[i] <threshold && relation_tph[i] >= threshold)
			{
				relation_type[i] = 2;
			}
			else if (relation_hpt[i] >= threshold && relation_tph[i] < threshold)
			{
				relation_type[i] = 3;
			}
			else
			{
				relation_type[i] = 4;
			}
		}

		//load_type(dataset.base_dir + "newType.txt");
		load_type(dataset.base_dir + "entity_type_418.txt");

		//load_relation_multi_type(dataset.base_dir + "head_tail_type_train_new.txt");
		load_relation_multi_type(dataset.base_dir + "head_tail_type_train_418.txt");

		load_relation_type(dataset.base_dir + "relation_specific.txt");

		for (auto i = data_train.begin(); i != data_train.end(); ++i)
		{
			count_hr[make_pair(i->first.first, i->second)]++;
			count_ht[make_pair(i->first.first, i->first.second)]++;
			count_rt[make_pair(i->second, i->first.second)]++;
		}

		cout << "triple_count" << endl;
	}
	DataModel(const Dataset& dataset, const string& file_zero_shot) :dataset(dataset)
	{
		load_training(dataset.base_dir + dataset.training);

		relation_hpt.resize(set_relation.size());
		relation_tph.resize(set_relation.size());
		for (auto i = 0; i != set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for (auto ds = rel_heads[i].begin(); ds != rel_heads[i].end(); ++ds)
			{
				++sum;
				total += ds->second.size();
			}
			relation_tph[i] = total / sum;
		}
		for (auto i = 0; i != set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for (auto ds = rel_tails[i].begin(); ds != rel_tails[i].end(); ++ds)
			{
				++sum;
				total += ds->second.size();
			}
			relation_hpt[i] = total / sum;
		}

		zeroshot_pointer = set_entity.size();
		load_testing(dataset.base_dir + dataset.developing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(dataset.base_dir + dataset.testing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(file_zero_shot, data_test_true, data_test_false, dataset.self_false_sampling);

		set_relation_head.resize(set_entity.size());
		set_relation_tail.resize(set_relation.size());
		prob_head.resize(set_entity.size());
		prob_tail.resize(set_entity.size());
		for (auto i = data_train.begin(); i != data_train.end(); ++i)
		{
			++prob_head[i->first.first];
			++prob_tail[i->first.second];

			++tails[i->second][i->first.first];
			++heads[i->second][i->first.second];

			set_relation_head[i->second].insert(i->first.first);
			set_relation_tail[i->second].insert(i->first.second);
		}

		for (auto & elem : prob_head)
		{
			elem /= data_train.size();
		}

		for (auto & elem : prob_tail)
		{
			elem /= data_train.size();
		}

		double threshold = 1.5;
		relation_type.resize(set_relation.size());
		for (auto i = 0; i < set_relation.size(); ++i)
		{
			if (relation_tph[i] < threshold && relation_hpt[i] < threshold)
			{
				relation_type[i] = 1;
			}
			else if (relation_hpt[i] < threshold && relation_tph[i] >= threshold)
			{
				relation_type[i] = 2;
			}
			else if (relation_hpt[i] >= threshold && relation_tph[i] < threshold)
			{
				relation_type[i] = 3;
			}
			else
			{
				relation_type[i] = 4;
			}
		}
	}
	void load_training(const string& filename)
	{
		int count = 0;
		fstream fin(filename.c_str());
		while (!fin.eof())
		{
			string head, tail, relation;

			fin >> head >> relation >> tail;

			if (head.empty())
			{
				continue;
			}
			if (entity_name_to_id.find(head) == entity_name_to_id.end())
			{
				entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
				entity_id_to_name.push_back(head);
			}

			if (entity_name_to_id.find(tail) == entity_name_to_id.end())
			{
				entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
				entity_id_to_name.push_back(tail);
			}

			if (relation_name_to_id.find(relation) == relation_name_to_id.end())
			{
				relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
				relation_id_to_name.push_back(relation);
			}

			data_train.push_back(make_pair(
				make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
				relation_name_to_id[relation]));

			check_data_train.insert(make_pair(
				make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
				relation_name_to_id[relation]));
			check_data_all.insert(make_pair(
				make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
				relation_name_to_id[relation]));

			set_entity.insert(head);
			set_entity.insert(tail);
			set_relation.insert(relation);

			++count_entity[head];
			++count_entity[tail];

			rel_heads[relation_name_to_id[relation]][entity_name_to_id[head]]
				.push_back(entity_name_to_id[tail]);
			rel_tails[relation_name_to_id[relation]][entity_name_to_id[tail]]
				.push_back(entity_name_to_id[head]);
			rel_finder[make_pair(entity_name_to_id[head], entity_name_to_id[tail])]
				= relation_name_to_id[relation];
			count++;
			if (relation_id_to_name.size()>1345)
			{
				cout << "aaaa" << relation_id_to_name.size() << endl;
				cout << count << endl;
			}
		}
		//cout << "aaaa" << relation_id_to_name.size() << endl;

		fin.close();
	}
	void load_testing(
		const string& filename,
		vector<pair<pair<int, int>, int>>& vin_true,
		vector<pair<pair<int, int>, int>>& vin_false,
		bool self_sampling = false)
	{
		fstream fin(filename.c_str());
		if (self_sampling == false)
		{
			while (!fin.eof())
			{
				string head, tail, relation;
				int flag_true;

				fin >> head >> relation >> tail;

				if (head.empty())
				{
					continue;
				}
				fin >> flag_true;

				if (entity_name_to_id.find(head) == entity_name_to_id.end())
				{
					entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
					entity_id_to_name.push_back(head);
				}

				if (entity_name_to_id.find(tail) == entity_name_to_id.end())
				{
					entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
					entity_id_to_name.push_back(tail);
				}

				if (relation_name_to_id.find(relation) == relation_name_to_id.end())
				{
					relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
					relation_id_to_name.push_back(relation);
				}

				set_entity.insert(head);
				set_entity.insert(tail);
				set_relation.insert(relation);

				if (flag_true == 1)
					vin_true.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));
				else
					vin_false.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));

				check_data_all.insert(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));
			}
		}
		else
		{
			while (!fin.eof())
			{
				string head, tail, relation;
				pair<pair<int, int>, int>	sample_false;

				fin >> head >> relation >> tail;
				if (head.empty())
				{
					continue;
				}
				if (entity_name_to_id.find(head) == entity_name_to_id.end())
				{
					entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
					entity_id_to_name.push_back(head);
				}

				if (entity_name_to_id.find(tail) == entity_name_to_id.end())
				{
					entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
					entity_id_to_name.push_back(tail);
				}

				if (relation_name_to_id.find(relation) == relation_name_to_id.end())
				{
					relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
					relation_id_to_name.push_back(relation);
				}

				set_entity.insert(head);
				set_entity.insert(tail);
				set_relation.insert(relation);

				sample_false_triplet(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]), sample_false);

				vin_true.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));
				vin_false.push_back(sample_false);

				check_data_all.insert(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));
			}
		}

		fin.close();
	}
	void load_type(const string& filename)
	{
		cout << "load entity type" << endl;

		fstream fin(filename.c_str());
		string line;
		while (!fin.eof())
		{
			getline(fin, line);
			if (line.empty())
			{
				continue;
			}
			string head, tail, relation;
			vector<string> sp_line;
			boost::split(sp_line, line, boost::is_any_of(" "));

			if (entity_name_to_id.find(sp_line[0]) == entity_name_to_id.end())
			{
				entity_name_to_id.insert(make_pair(sp_line[0], entity_name_to_id.size()));
				entity_id_to_name.push_back(sp_line[0]);
			}

			for (size_t i = 1; i < sp_line.size(); i++)
			{
				if (type_name_to_id.find(sp_line[0]) == type_name_to_id.end())
				{
					type_name_to_id.insert(make_pair(sp_line[i], type_name_to_id.size()));
					type_id_to_name.push_back(sp_line[i]);
					type_of_entity[entity_name_to_id[sp_line[0]]].insert(type_name_to_id[sp_line[i]]);

					type2entity[type_name_to_id[sp_line[i]]].push_back(entity_name_to_id[sp_line[0]]);
				}
			}
		}
		fin.close();
	}
	void load_relation_multi_type(const string& filename)
	{
		cout << "load relation multi type" << endl;

		type_of_relation.clear();
		//load_relation_type(dataset.base_dir + "relation_specific.txt");
		//cout << "be" << endl;
		cout << filename.c_str() << endl;
		fstream fin(filename.c_str());
		string line;
		//cout << "af" << endl;
		int curent_relation;
		while (!fin.eof())
		{
			getline(fin, line);
			if (line.empty())
			{
				continue;
			}
			//cout << line << endl;
			if (line.substr(0, 4) == "rela")
			{
				//cout << line.substr(10, line.length() - 10) << endl;
				curent_relation = relation_name_to_id[line.substr(10, line.length() - 10)];
				//cout << curent_relation << endl;
				getline(fin, line);
				while (line.substr(0, 4) != "  ta")
				{
					//cout << line << endl;
					//cout << type_name_to_id[line.substr(4, line.length() - 4)] << endl;
					type_of_relation[curent_relation].first.insert(type_name_to_id[line.substr(4, line.length() - 4)]);
					getline(fin, line);
				}
				getline(fin, line);
				while (line.substr(0, 4) != "----")
				{
					//cout << line << endl;
					//cout << type_name_to_id[line.substr(4, line.length() - 4)] << endl;
					type_of_relation[curent_relation].second.insert(type_name_to_id[line.substr(4, line.length() - 4)]);
					getline(fin, line);
				}
			}
			if (relation_id_to_name.size() > 1345)
			{
				cout << "aaaa" << endl;
				int ttt;
				cin >> ttt;
			}
		}
		fin.close();
	}
	void load_relation_type(const string& filename)
	{
		cout << "load relation single type" << endl;

		//type_of_relation.clear();
		//cout << "be" << endl;
		single_type_of_relation.clear();
		fstream fin(filename.c_str());
		string line;
		//cout << "af" << endl;
		int curent_relation;
		string relation, head_type, tail_type;
		while (!fin.eof())
		{
			fin >> relation >> head_type >> tail_type;
			if (relation.empty())
			{
				continue;
			}
			//type_of_relation[relation_name_to_id[relation]].first.insert(type_name_to_id[head_type]);
			//type_of_relation[relation_name_to_id[relation]].second.insert(type_name_to_id[tail_type]);

			single_type_of_relation[relation_name_to_id[relation]] = make_pair(type_name_to_id[head_type], type_name_to_id[tail_type]);

			if (relation_id_to_name.size() > 1345)
			{
				cout << "qqqq" << endl;
				int ttt;
				cin >> ttt;
			}
		}
		fin.close();

	}

	void calc_rel_ent_pos_training_typescore()
	{
		rel_ent_pos_typescore.clear();
		vector<int> temp(200);
		vector<int>::iterator last_it;
		rel_ent_pos_typescore.resize(relation_id_to_name.size());
		for (size_t i = 0; i < relation_id_to_name.size(); i++)
		{
			rel_ent_pos_typescore[i].resize(entity_id_to_name.size());
			for (size_t j = 0; j < entity_id_to_name.size(); j++)
			{
				rel_ent_pos_typescore[i][j].resize(2);
			}
		}

		//#pragma omp parallel for
		for (size_t i = 0; i < relation_id_to_name.size(); i++)
		{
			double rel_fir_size = type_of_relation[i].first.size();
			double rel_sec_size = type_of_relation[i].second.size();
			//cout << entity_id_to_name.size() << endl;
			//rel_ent_pos_typescore[i].resize(entity_id_to_name.size());
			for (size_t j = 0; j < entity_id_to_name.size(); j++)
			{
				//rel_ent_pos_typescore[i][j].resize(2);
				last_it = std::set_intersection(
					type_of_entity[j].begin(),
					type_of_entity[j].end(),
					type_of_relation[i].first.begin(),
					type_of_relation[i].first.end(),
					temp.begin());
				rel_ent_pos_typescore[i][j][0] = (1.0 + (double)(last_it - temp.begin())) / ((double)rel_fir_size);
				if (rel_ent_pos_typescore[i][j][0] <0.01)
				{
					cout << rel_ent_pos_typescore[i][j][0] << endl;
				}
				//if (rel_ent_pos_typescore[i][j][0] < 0.4)
				//{
				//	rel_ent_pos_typescore[i][j][0] = 1e-4;
				//}
				//else {
				//	rel_ent_pos_typescore[i][j][0] = 1.0;
				//}
				//rel_ent_pos_typescore[i][j][0] = 1.0 / rel_ent_pos_typescore[i][j][0];
				//rel_ent_pos_typescore[i][j][0] = 1.0;

				//cout << "sd " << si << "  " << rel_ent_pos_typescore[i][j][0] << endl;
				double prob_simu_t = rel_ent_pos_typescore[i][j][0];
				//if (prob_simu_t < 1.1 && prob_simu_t > -0.1)
				//{
				//	//cout << prob_simu_t << endl;

				//}
				//else{
				//	cout << prob_simu_t << endl;
				//	cout << i << " " << j << " " << "0" << " " << last_it - temp.begin() << " " << rel_fir_size << endl;

				//	int kk;
				//	cin >> kk;
				//}

				last_it = std::set_intersection(
					type_of_entity[j].begin(),
					type_of_entity[j].end(),
					type_of_relation[i].second.begin(),
					type_of_relation[i].second.end(),
					temp.begin());
				rel_ent_pos_typescore[i][j][1] = (1.0 + (double)(last_it - temp.begin())) / ((double)rel_sec_size);
				if (rel_ent_pos_typescore[i][j][1] <0.01)
				{
					cout << rel_ent_pos_typescore[i][j][0] << endl;
				}
				//if (rel_ent_pos_typescore[i][j][1] < 0.4)
				//{
				//	rel_ent_pos_typescore[i][j][1] = 1e-4;
				//}
				//else {
				//	rel_ent_pos_typescore[i][j][1] = 1.0;
				//}
				//rel_ent_pos_typescore[i][j][1] = 1.0 / rel_ent_pos_typescore[i][j][1];
				//rel_ent_pos_typescore[i][j][1] = 1.0;

				//cout << "sd " << si << "  " << rel_sec_size << "  " << rel_ent_pos_typescore[i][j][1] << endl;
				prob_simu_t = rel_ent_pos_typescore[i][j][1];
				//if (prob_simu_t < 1.1 && prob_simu_t > -0.1)
				//{
				//	//cout << prob_simu_t << endl;

				//}
				//else{
				//	cout << prob_simu_t << endl;
				//	cout << i << " " << j << " " << "1" << " " << last_it - temp.begin() << " " << rel_sec_size << " " << relation_id_to_name[i] << " " << entity_id_to_name[j] << endl;
				//	int kk;
				//	cin >> kk;
				//}

			}
		}
	}

	void calc_rel_ent_pos_typescore()
	{
		rel_ent_pos_typescore.clear();
		vector<int> head_temp(200);
		vector<int> tail_temp(200);
		vector<int> type_union;
		vector<int>::iterator head_last, tail_last;


		rel_ent_pos_typescore.resize(relation_id_to_name.size());
		for (size_t i = 0; i < relation_id_to_name.size(); i++)
		{
			rel_ent_pos_typescore[i].resize(entity_id_to_name.size());
			for (size_t j = 0; j < entity_id_to_name.size(); j++)
			{
				rel_ent_pos_typescore[i][j].resize(2);
			}
		}
		vector<double> head_v;
		vector<double> neg_head_v;
		vector<double> tail_v;
		vector<double> neg_tail_v;

		head_v.resize(entity_id_to_name.size());
		neg_head_v.resize(entity_id_to_name.size());
		tail_v.resize(entity_id_to_name.size());
		neg_tail_v.resize(entity_id_to_name.size());

		//#pragma omp parallel for
		for (size_t i = 0; i < relation_id_to_name.size(); i++)
		{
			double rel_fir_size = type_of_relation[i].first.size();
			double rel_sec_size = type_of_relation[i].second.size();
			double head_prob = 0.0;
			double head_neg_prob = 0.0;
			double tail_prob = 0.0;
			double tail_neg_prob = 0.0;

			int head_num = 0;
			int head_neg_num = 0;
			int tail_num = 0;
			int tail_neg_num = 0;


			for (size_t j = 0; j < entity_id_to_name.size(); j++)
			{
				//取head_type交集
				head_last = std::set_intersection(
					type_of_entity[j].begin(),
					type_of_entity[j].end(),
					type_of_relation[i].first.begin(),
					type_of_relation[i].first.end(),
					head_temp.begin());

				//取tail_type并集
				/*		std::set_union(type_of_entity[j].begin(),
				type_of_entity[j].end(),
				type_of_relation[i].first.begin(),
				type_of_relation[i].first.end(),
				inserter(type_union, type_union.begin()));*/

				rel_ent_pos_typescore[i][j][0] = ((double)(head_last - head_temp.begin() - 0.99)) / ((double)rel_fir_size);

				//rel_ent_pos_typescore[i][j][0] = ((double)(last_it - temp.begin())) / ((double)type_union.size());

				if (rel_heads[i][j].size() > 0){
					head_prob += rel_ent_pos_typescore[i][j][0];
					head_v[head_num] = rel_ent_pos_typescore[i][j][0];

					head_num++;

					//cout << "head:  " << rel_ent_pos_typescore[i][j][0] << endl;
					//cout << "right head:  " << rel_ent_pos_typescore[i][j][0] << endl;
				}
				else{
					head_neg_prob += rel_ent_pos_typescore[i][j][0];
					neg_head_v[head_neg_num] = rel_ent_pos_typescore[i][j][0];

					head_neg_num++;
					//cout << "wrong head:  " << rel_ent_pos_typescore[i][j][0] << endl;
				}
				//if (rel_ent_pos_typescore[i][j][0] < 0.7)
				//{
				//	rel_ent_pos_typescore[i][j][0] = 1e-4;
				//}
				//else{
				//	rel_ent_pos_typescore[i][j][0] = 1;
				//}

				//取tail_type的交集
				tail_last = std::set_intersection(
					type_of_entity[j].begin(),
					type_of_entity[j].end(),
					type_of_relation[i].second.begin(),
					type_of_relation[i].second.end(),
					tail_temp.begin());

				//取tail_type的并集
				/*	std::set_union(type_of_entity[j].begin(),
				type_of_entity[j].end(),
				type_of_relation[i].second.begin(),
				type_of_relation[i].second.end(),
				inserter(type_union, type_union.begin()));*/

				rel_ent_pos_typescore[i][j][1] = ((double)(tail_last - tail_temp.begin() - 0.99)) / ((double)rel_sec_size);
				//rel_ent_pos_typescore[i][j][1] = ((double)(last_it - temp.begin())) / ((double)type_union.size());

				if (rel_tails[i][j].size() > 0){
					tail_prob += rel_ent_pos_typescore[i][j][1];
					tail_v[tail_num] = rel_ent_pos_typescore[i][j][1];

					tail_num++;
					//cout << "right tail:  " << rel_ent_pos_typescore[i][j][1] << endl;
				}
				else{
					tail_neg_prob += rel_ent_pos_typescore[i][j][1];
					neg_tail_v[tail_neg_num] = rel_ent_pos_typescore[i][j][1];

					tail_neg_num++;
					//cout << "wrong tail:  " << rel_ent_pos_typescore[i][j][1] << endl;
				}
				//cout << "rel=" << relation_id_to_name[i] << ", entity=" << entity_id_to_name[j] << ", pos=1" << endl;
				//cout << head_last - head_temp.begin() << " " << rel_fir_size << " " <<  rel_ent_pos_typescore[i][j][0] << " " << tail_last - tail_temp.begin() << " " << rel_sec_size << " " << rel_ent_pos_typescore[i][j][1] << endl;
				/*if (rel_ent_pos_typescore[i][j][1] <0.01)
				{
				cout << rel_ent_pos_typescore[i][j][0] << endl;
				}*/
				/*			if (rel_ent_pos_typescore[i][j][1] < 0.7)
				{
				rel_ent_pos_typescore[i][j][1] = 1e-4;
				}*/
				/*else{
				rel_ent_pos_typescore[i][j][1] = 1;
				}*/
				//cout << "average prob: " << rel_ent_pos_typescore[i][j][0] << " " << rel_ent_pos_typescore[i][j][1]<<" "<<num << " " << prob / num << "  " << neg_num << " " << neg_prob / neg_num << " " << (prob + neg_prob) / (num + neg_num) << endl;
				//cout << head_last - head_temp.begin() << " " << rel_fir_size << " " << rel_ent_pos_typescore[i][j][0] << " " << tail_last - tail_temp.begin() << " " << rel_sec_size << " " << rel_ent_pos_typescore[i][j][1] << endl;
				head_temp.clear();
				tail_temp.clear();
				type_union.clear();

			}
			//sort(head_v.begin(), head_v.end(), greater<double>());
			//sort(tail_v.begin(), tail_v.end(), greater<double>());
			//sort(neg_head_v.begin(), neg_head_v.end(), greater<double>());
			//sort(neg_tail_v.begin(), neg_tail_v.end(), greater<double>());
			//
			//double head_pro=0.0;
			//double neg_head_pro = 0.0;
			//double tail_pro = 0.0;
			//double neg_tail_pro = 0.0;
			//
			//int count = 50;
			//
			//int num_head = head_num > count ? count : head_num;
			//int neg_num_head = head_neg_num > count ? count : head_neg_num;

			//int num_tail = tail_num > count ? count : tail_num;
			//int neg_num_tail = tail_neg_num> count ? count : tail_neg_num;
			//
			//for (int i = 0; i < num_head; i++){
			//	head_pro += head_v[i];
			//}

			//for (int i = 0; i < neg_num_head; i++){
			//	neg_head_pro += neg_head_v[i];
			//}

			//for (int i = 0; i < num_tail; i++){
			//	tail_pro += tail_v[i];
			//}

			//for (int i = 0; i < neg_num_tail; i++){
			//	neg_tail_pro += neg_tail_v[i];
			//}

			//head_v.clear();
			//tail_v.clear();
			//neg_head_v.clear();
			//neg_tail_v.clear();

			//cout << "-------" << endl;
			////cout << i << " " << head_num << " " << head_neg_num << " " << tail_num << " " << tail_neg_num << endl;
			//
			//cout << "average prob: " << i << " " << head_pro / num_head << "  " << neg_head_pro / neg_num_head << " " << tail_pro / num_tail << "  " << neg_tail_pro / neg_num_tail << endl;
			//cout << "average prob: " <<i<<" "<< head_prob / head_num << "  " << head_neg_prob / head_neg_num << " " << tail_prob / tail_num << "  " << tail_neg_prob / tail_neg_num << endl;
			//cout << "-------" << endl;
		}
	}

	void sample_false_triplet_unif(
		const pair<pair<int, int>, int>& origin,
		pair<pair<int, int>, int>& triplet) const
	{

		double prob = 0.5;

		triplet = origin;
		while (true)
		{
			if (rand() % 1000 < 1000 * prob)
			{
				triplet.first.second = rand() % set_entity.size();
			}
			else
			{
				triplet.first.first = rand() % set_entity.size();
			}

			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}

	void sample_false_triplet(
		const pair<pair<int, int>, int>& origin,
		pair<pair<int, int>, int>& triplet) const
	{

		double prob = relation_hpt[origin.second] / (relation_hpt[origin.second] + relation_tph[origin.second]);

		triplet = origin;
		while (true)
		{
			if (rand() % 1000 < 1000 * prob)
			{
				triplet.first.second = rand() % set_entity.size();
			}
			else
			{
				triplet.first.first = rand() % set_entity.size();
			}

			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}

	void sample_false_triplet_relation(
		const pair<pair<int, int>, int>& origin,
		pair<pair<int, int>, int>& triplet) const
	{

		double prob = relation_hpt[origin.second] / (relation_hpt[origin.second] + relation_tph[origin.second]);
		//double prob = 0.5;
		triplet = origin;
		while (true)
		{
			if (rand() % 100 < 50)
				triplet.second = rand() % set_relation.size();
			else if (rand() % 1000 < 1000 * prob)
			{
				triplet.first.second = rand() % set_entity.size();
			}
			else
			{
				triplet.first.first = rand() % set_entity.size();
			}

			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}

	void sample_false_triplet_rela(
		const pair<pair<int, int>, int>& origin,
		pair<pair<int, int>, int>& triplet) const
	{
		triplet = origin;
		while (true)
		{
			//if (rand() % 100 < 50)
			triplet.second = rand() % set_relation.size();
			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}

	void sample_false_triplet_type(const pair<pair<int, int>, int>& origin,
		pair<pair<int, int>, int>& triplet){

		triplet = origin;

		int tail_type = single_type_of_relation[triplet.second].second;
		int head_type = single_type_of_relation[triplet.second].first;


		if (rand() % 1000 < 500){  //update tail

			if (type2entity[tail_type].size()>1){ //存在待选实体(去除本身)
				while (triplet.first.second == origin.first.second)
					triplet.first.second = type2entity[tail_type][rand() % type2entity[tail_type].size()];
			}
			else{ //不存在待选实体(随机替换)
				while (check_data_train.find(triplet) != check_data_train.end())
					triplet.first.second = rand() % set_entity.size();
			}
		}
		else{   //update head

			//type entity1;entity2;entity3;entity4
			if (type2entity[head_type].size()>1){//存在待选实体(去除本身)
				while (triplet.first.first == origin.first.first)
					triplet.first.first = type2entity[head_type][rand() % type2entity[head_type].size()];
			}
			else{//不存在待选实体(随机替换)
				while (check_data_train.find(triplet) != check_data_train.end())
					triplet.first.first = rand() % set_entity.size();
			}
		}
	}
};

class ModelLogging
{
protected:
	ofstream fout;
public:
	ModelLogging(const string& base_dir)
	{
		const time_t log_time = time(nullptr);
		struct tm* current_time = localtime(&log_time);
		stringstream ss;
		ss << 1900 + current_time->tm_year << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mon + 1 << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mday << " ";
		ss << setfill('0') << setw(2) << current_time->tm_hour << ".";
		ss << setfill('0') << setw(2) << current_time->tm_min << ".";
		ss << setfill('0') << setw(2) << current_time->tm_sec;
		fout.open((base_dir + ss.str() + ".log").c_str());
		fout << '[' << ss.str() << ']' << '\t' << "Starting...";
	}
	ModelLogging& record()
	{
		const time_t log_time = time(nullptr);
		struct tm* current_time = localtime(&log_time);
		stringstream ss;
		ss << 1900 + current_time->tm_year << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mon + 1 << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mday << " ";
		ss << setfill('0') << setw(2) << current_time->tm_hour << ".";
		ss << setfill('0') << setw(2) << current_time->tm_min << ".";
		ss << setfill('0') << setw(2) << current_time->tm_sec;
		fout << endl;
		fout << '[' << ss.str() << ']' << '\t';
		return *this;
	}
	template<typename T>
	ModelLogging& operator << (T things)
	{
		fout << things;
		return *this;
	}

	~ModelLogging()
	{
		fout << endl;
		fout.close();
	}
};
class TransT
{
private:
	int epos = 0;
	double loss = 0.0;

public:
	vector<vec>				embedding_relation;
	vector<vector<vec>>		embedding_clusters;
	vector<vec>				weights_clusters;
	vector<int>				size_clusters;
	vector<vector<int> >	type_of_vector;
	map<int, pair<set<int>, set<int> > > rel_type;
	map<int, set<int> > ent_type;
	vector<vector<vector<double> > > rel_simu;

	const int				n_cluster;
	const double			alpha;
	const bool				single_or_total;
	const double			training_threshold;
	const int				dim;
	const bool				be_weight_normalized;
	const int				step_before;
	const double			normalizor;
	int                     new_entity;
	const int				max_cluster_size;

	double					CRP_factor;

	DataModel&	data_model;
	TaskType		task_type;
	const bool			be_deleted_data_model;
	string logging_base_path;


	ModelLogging&		logging;
	void log_better_entity(const TaskType task, const pair<pair<int, int>, int>& triplet){};

	TransT(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		int n_cluster,
		int max_cluster_size,
		double CRP_factor,
		int step_before = 10,
		bool sot = false,
		bool be_weight_normalized = true)
		:data_model(*(new DataModel(dataset))), task_type(task_type), logging_base_path(logging_base_path), dim(dim), alpha(alpha),
		training_threshold(training_threshold), n_cluster(n_cluster), max_cluster_size(max_cluster_size), CRP_factor(CRP_factor),
		single_or_total(sot), be_weight_normalized(be_weight_normalized), step_before(step_before),
		normalizor(1.0 / pow(3.1415, dim / 2)), logging(*(new ModelLogging(logging_base_path))), be_deleted_data_model(true)
	{
		printf("record start \n");
		logging.record() << "\t[Name]\tTransT";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Cluster Counts]\t" << n_cluster;
		logging.record() << "\t[CRP Factor]\t" << CRP_factor;
		if (be_weight_normalized)
			logging.record() << "\t[Weight Normalized]\tTrue";
		else
			logging.record() << "\t[Weight Normalized]\tFalse";
		if (sot)
			logging.record() << "\t[Single or Total]\tTrue";
		else
			logging.record() << "\t[Single or Total]\tFalse";
		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = randu(dim, 1); });
		embedding_clusters.resize(count_entity());
		for (auto &elem_vec : embedding_clusters)
		{
			elem_vec.resize(30);
			for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem){elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
		}
		type_of_vector.resize(count_entity());
		for (auto &elem_vec : type_of_vector)
		{
			elem_vec.resize(30);
			for_each(elem_vec.begin(), elem_vec.end(), [=](int& elem){elem = -1; });
		}
		weights_clusters.resize(count_entity());
		for (auto & elem_vec : weights_clusters)
		{
			elem_vec.resize(30);
			elem_vec.fill(0.0);
			for (auto i = 0; i<n_cluster; ++i)
			{
				elem_vec[i] = 1.0 / n_cluster;
			}
		}
		size_clusters.resize(count_entity(), n_cluster);
		new_entity = 0;
		rel_type = data_model.type_of_relation;
		ent_type = data_model.type_of_entity;
		calc_rel_simu();
		//cout << "init " << endl;
	}
public:
	~TransT()
	{
		logging.record();
		if (be_deleted_data_model)
		{
			delete &data_model;
			delete &logging;
		}
	}
public: // auxiliary functions
	string Loss_path;
	void setPath(string path){
		Loss_path = path;
	}
	void update_loss(double lo) {
		//if (epos%100==0)
		loss += lo;
	}
	int get_new_entity()
	{
		return new_entity;
	}
	double prob_simu(int rel, int ent, int pos)
	{
		if (pos > 2 || pos < 1) return -1.0;
		return data_model.rel_ent_pos_typescore[rel][ent][pos - 1];
	}
	inline double sign1(const double& x)
	{
		if (x == 0)
			return 0;
		else
			return x>0 ? +1 : -1;
	}
	int count_entity() const
	{
		return data_model.set_entity.size();
	}
	int count_relation() const
	{
		return data_model.set_relation.size();
	}
	const DataModel& get_data_model() const
	{
		return data_model;
	}
public:
	void calc_rel_simu()
	{
		rel_simu.clear();

		rel_simu.resize(count_relation());
		for (size_t i = 0; i < count_relation(); i++)
		{
			rel_simu[i].resize(count_relation());
			for (size_t j = 0; j < count_relation(); j++)
			{
				rel_simu[i][j].resize(2);
			}
		}
#pragma omp parallel for
		for (size_t i = 0; i < count_relation(); i++)
		{
			double rel_fir_size = data_model.type_of_relation[i].first.size();
			double rel_sec_size = data_model.type_of_relation[i].second.size();
			//cout << entity_id_to_name.size() << endl;
			//rel_ent_pos_typescore[i].resize(entity_id_to_name.size());
			for (size_t j = 0; j < count_relation(); j++)
			{
				//rel_ent_pos_typescore[i][j].resize(2);
				vector<int> temp(200);
				vector<int>::iterator last_it;
				last_it = std::set_intersection(
					data_model.type_of_relation[j].first.begin(),
					data_model.type_of_relation[j].first.end(),
					data_model.type_of_relation[i].first.begin(),
					data_model.type_of_relation[i].first.end(),
					temp.begin());
				rel_simu[i][j][0] = ((double)(last_it - temp.begin())) / ((double)rel_fir_size);

				last_it = std::set_intersection(
					data_model.type_of_relation[j].second.begin(),
					data_model.type_of_relation[j].second.end(),
					data_model.type_of_relation[i].second.begin(),
					data_model.type_of_relation[i].second.end(),
					temp.begin());
				rel_simu[i][j][1] = ((double)(last_it - temp.begin())) / ((double)rel_sec_size);

			}
		}
	}
public: // train functions
	double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		if (single_or_total == false)
			return training_prob_triplets(triplet);
		double	mixed_prob = 1e-20;
		for (int c = 0; c<size_clusters[triplet.first.first]; ++c)
		for (int d = 0; d < size_clusters[triplet.first.second]; ++d)
		{
			vec error_c = (embedding_relation[triplet.second] + embedding_clusters[triplet.first.first][c]
				- embedding_clusters[triplet.first.second][d]);
			mixed_prob = max(mixed_prob, fabs(weights_clusters[triplet.first.first][c] * weights_clusters[triplet.first.second][d])
				* exp(-sum(abs(error_c))));
		}
		return mixed_prob;
	}
	double training_prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		double	mixed_prob = 1e-20;
		for (int c = 0; c<size_clusters[triplet.first.first]; ++c)
		for (int d = 0; d < size_clusters[triplet.first.second]; ++d)
		{
			vec error_c = (embedding_relation[triplet.second] + embedding_clusters[triplet.first.first][c]
				- embedding_clusters[triplet.first.second][d]);
			mixed_prob += fabs(weights_clusters[triplet.first.first][c] * weights_clusters[triplet.first.second][d])
				* exp(-sum(abs(error_c)));
		}

		return mixed_prob;
	}
	void train_cluster_once(const pair<pair<int, int>, int>& triplet, int flag, int cluster, int cluster_, double prob, double factor)
	{
		vec& head = embedding_clusters[triplet.first.first][cluster];
		vec& tail = embedding_clusters[triplet.first.second][cluster_];
		vec& relation = embedding_relation[triplet.second];
		double prob_local_true = exp(-sum(abs(head + relation - tail)));
		weights_clusters[triplet.first.first][cluster] += flag *
			alpha / prob * prob_local_true  * fabs(weights_clusters[triplet.first.second][cluster_]) *
			sign1(weights_clusters[triplet.first.first][cluster]);
		weights_clusters[triplet.first.second][cluster_] += flag *
			alpha / prob * prob_local_true * fabs(weights_clusters[triplet.first.first][cluster]) *
			sign1(weights_clusters[triplet.first.second][cluster_]);
		head -= flag * alpha * sign(head + relation - tail)
			* prob_local_true / prob * fabs(weights_clusters[triplet.first.first][cluster] *
			weights_clusters[triplet.first.second][cluster_]);
		tail += flag * alpha * sign(head + relation - tail)
			* prob_local_true / prob * fabs(weights_clusters[triplet.first.first][cluster] *
			weights_clusters[triplet.first.second][cluster_]);
		relation -= flag * alpha * sign(head + relation - tail)
			* prob_local_true / prob * fabs(weights_clusters[triplet.first.first][cluster] *
			weights_clusters[triplet.first.second][cluster_]);
		if (norm(head, 2) > 1.0)
			head = normalise(head);
		if (norm(tail, 2) > 1.0)
			tail = normalise(tail);
	}
	void train_triplet(const pair<pair<int, int>, int>& triplet, const pair<pair<int, int>, int>& triplet_f)
	{
		vector<vec>& head = embedding_clusters[triplet.first.first];
		vector<vec>& tail = embedding_clusters[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double prob_true = training_prob_triplets(triplet);
		double prob_false = training_prob_triplets(triplet_f);
		if (prob_true / prob_false > exp(training_threshold))
			return;
		update_loss((-log(prob_true) + log(prob_false) + training_threshold));
		//update positive sample
		for (int c = 0; c<size_clusters[triplet.first.first]; ++c)
		for (int d = 0; d < size_clusters[triplet.first.second]; ++d)
		{
			train_cluster_once(triplet, 1, c, d, prob_true, alpha);
		}
		//update negative sample
		for (int c = 0; c<size_clusters[triplet_f.first.first]; ++c)
		for (int d = 0; d < size_clusters[triplet_f.first.second]; ++d)
		{
			train_cluster_once(triplet_f, -1, c, d, prob_false, alpha);
		}
		double prob_new_component = CRP_factor * exp(-sum(abs(relation)));
		double prob_new = prob_new_component / (prob_new_component + prob_true);
		if (((double)rand()) / RAND_MAX < prob_new && epos >= step_before)
		{
#pragma omp critical
			{
				double max_weight_value;
				int max_weight_pos;
				bool flag;
				if (((double)rand()) / RAND_MAX < 0.5) {
					flag = true;
					for (size_t vi = 0; vi < size_clusters[triplet.first.second]; vi++)
					{
						if (type_of_vector[triplet.first.second][vi] == -1)
						{
							continue;
						}
						if (rel_simu[triplet.second][type_of_vector[triplet.first.second][vi]][1] > 0.7)
						{
							flag = false;
							break;
						}
					}
					if (flag && size_clusters[triplet.first.second] <= 10) {
						new_entity++;
						weights_clusters[triplet.first.second][size_clusters[triplet.first.second]] = CRP_factor;
						embedding_clusters[triplet.first.second][size_clusters[triplet.first.second]] = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim);
						type_of_vector[triplet.first.second][size_clusters[triplet.first.second]] = triplet.second;
						++size_clusters[triplet.first.second];
					}
				}
				else {
					flag = true;
					for (size_t vi = 0; vi < size_clusters[triplet.first.first]; vi++)
					{
						if (type_of_vector[triplet.first.first][vi] == -1)
						{
							continue;
						}
						if (rel_simu[triplet.second][type_of_vector[triplet.first.first][vi]][0] > 0.7)
						{
							flag = false;
							break;
						}
					}
					if (flag && size_clusters[triplet.first.first] <= 10) {
						new_entity++;
						weights_clusters[triplet.first.first][size_clusters[triplet.first.first]] = CRP_factor;
						embedding_clusters[triplet.first.first][size_clusters[triplet.first.first]] = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim);
						type_of_vector[triplet.first.first][size_clusters[triplet.first.first]] = triplet.second;
						++size_clusters[triplet.first.first];
					}
				}
			}
		}
		vec& relation_f = embedding_relation[triplet_f.second];
		if (norm(relation, 2) > 1.0)
			relation = normalise(relation);
		if (norm(relation_f, 2) > 1.0)
			relation_f = normalise(relation_f);
		if (be_weight_normalized) {
			weights_clusters[triplet.first.first] = normalise(weights_clusters[triplet.first.first]);
			weights_clusters[triplet.first.second] = normalise(weights_clusters[triplet.first.second]);
		}
	}
	void run(int total_epos)
	{
		//cout << "start train run" << endl;
		logging.record() << "\t[Epos]\t" << total_epos;
		ofstream fout(Loss_path);
		epos = 0;
		--total_epos;
		boost::progress_display	cons_bar(total_epos);
		while (epos < total_epos)
		{
			epos++;
			++cons_bar;
			loss = 0.0;
#pragma omp parallel for
			for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
			{
				pair<pair<int, int>, int> triplet_f;
				pair<pair<int, int>, int> triplet_rel;
				//data_model.sample_false_triplet_relation(*i, triplet_f);
				data_model.sample_false_triplet_unif(*i, triplet_f);
				//data_model.sample_false_triplet_rela(*i, triplet_f);
				train_triplet(*i, triplet_f);
			}
			printf("epos=%d   new_entity=%d   loss=%.16lf\n", epos, get_new_entity(), loss);
		}
		fout.close();
	}
public:
	double		best_triplet_result;
	double		best_link_mean;
	double		best_link_hitatten;
	double		best_link_fmean;
	double		best_link_fhitatten;
	void changeTaskType(const TaskType& task_type) {
		this->task_type = task_type;
	}
public: // test functions
	void test_triplet_classification()
	{
		//data_model.load_relation_multi_type(data_model.dataset.base_dir + "head_tail_type_train_new.txt");
		//data_model.calc_rel_ent_pos_typescore();
		double real_hit = 0;
		for (auto r = 0; r < data_model.set_relation.size(); ++r)
		{
			vector<pair<double, bool>>	threshold_dev;
			for (auto i = data_model.data_dev_true.begin(); i != data_model.data_dev_true.end(); ++i)
			{
				/*pair<pair<int, int>, int> t = *i;
				double typescore = data_model.rel_ent_pos_typescore[t.second][t.first.first][0] * data_model.rel_ent_pos_typescore[t.second][t.first.second][1];*/
				if (i->second != r)
					continue;
				//threshold_dev.push_back(make_pair(prob_triplets(*i)*typescore, true));
				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for (auto i = data_model.data_dev_false.begin(); i != data_model.data_dev_false.end(); ++i)
			{
				/*pair<pair<int, int>, int> t = *i;
				double typescore = data_model.rel_ent_pos_typescore[t.second][t.first.first][0] * data_model.rel_ent_pos_typescore[t.second][t.first.second][1];*/
				if (i->second != r)
					continue;
				//threshold_dev.push_back(make_pair(prob_triplets(*i)*typescore, false));
				threshold_dev.push_back(make_pair(prob_triplets(*i), false));
			}
			sort(threshold_dev.begin(), threshold_dev.end());
			double threshold;
			double vari_mark = 0;
			int total = 0;
			int hit = 0;
			for (auto i = threshold_dev.begin(); i != threshold_dev.end(); ++i)
			{
				if (i->second == false)
					++hit;
				++total;

				if (vari_mark <= 2 * hit - total + data_model.data_dev_true.size())
				{
					vari_mark = 2 * hit - total + data_model.data_dev_true.size();
					threshold = i->first;
				}
			}
			double lreal_hit = 0;
			double lreal_total = 0;
			for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;
				if (prob_triplets(*i) > threshold)
					++real_hit, ++lreal_hit;
			}
			for (auto i = data_model.data_test_false.begin(); i != data_model.data_test_false.end(); ++i)
			{
				if (i->second != r)
					continue;
				++lreal_total;
				if (prob_triplets(*i) <= threshold)
					++real_hit, ++lreal_hit;
			}
		}
		std::cout << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size());
		best_triplet_result = max(
			best_triplet_result,
			real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size()));
		std::cout << ", Best = " << best_triplet_result << endl;
		logging.record() << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size())
			<< ", Best = " << best_triplet_result;
		std::cout.flush();
	}
	void test_true_triplet_classification()
	{
		double real_hit = 0;
		for (auto r = 0; r < data_model.set_relation.size(); ++r)
		{
			vector<pair<double, bool>>	threshold_dev;
			for (auto i = data_model.data_dev_true.begin(); i != data_model.data_dev_true.end(); ++i)
			{
				if (i->second != r)
					continue;
				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for (auto i = data_model.data_dev_false.begin(); i != data_model.data_dev_false.end(); ++i)
			{
				if (i->second != r)
					continue;
				threshold_dev.push_back(make_pair(prob_triplets(*i), false));
			}
			sort(threshold_dev.begin(), threshold_dev.end());
			double threshold;
			double vari_mark = 0;
			int total = 0;
			int hit = 0;
			for (auto i = threshold_dev.begin(); i != threshold_dev.end(); ++i)
			{
				if (i->second == false)
					++hit;
				++total;
				if (vari_mark <= 2 * hit - total + data_model.data_dev_true.size())
				{
					vari_mark = 2 * hit - total + data_model.data_dev_true.size();
					threshold = i->first;
				}
			}
			double lreal_hit = 0;
			double lreal_total = 0;
			for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
			{
				if (i->second != r)
					continue;
				++lreal_total;
				if (prob_triplets(*i) > threshold) {
					++real_hit, ++lreal_hit;
					//cout << "p=" << prob_triplets(*i) << " threshold=" << threshold << endl;
				}
			}
			//for (auto i = data_model.data_test_false.begin(); i != data_model.data_test_false.end(); ++i)
			//{
			//	if (i->second != r)
			//		continue;

			//	++lreal_total;
			//	if (prob_triplets(*i) <= threshold)
			//		++real_hit, ++lreal_hit;
			//}
			//std::cout << epos << "\t Accuracy = " << real_hit / (data_model.data_test_true.size());
			//logging.record()<<data_model.relation_id_to_name.at(r)<<"\t"
			//	<<lreal_hit/lreal_total;
		}
		cout << real_hit << " " << data_model.data_test_true.size() << endl;
		std::cout << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size());
		best_triplet_result = max(
			best_triplet_result,
			real_hit / (data_model.data_test_true.size()));
		std::cout << ", Best = " << best_triplet_result << endl;
		logging.record() << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size())
			<< ", Best = " << best_triplet_result;
		std::cout.flush();
	}
	void test_link_prediction(int hit_rank = 10, const int part = 0)
	{
		test_link_prediction_multi_type(hit_rank, part);
		return;
		ofstream fout;
		if (task_type == LinkPredictionHead)
		{
			fout.open(logging_base_path + "head_rank_type.txt");
		}
		if (task_type == LinkPredictionTail)
		{
			fout.open(logging_base_path + "tail_rank.txt");
		}
		if (task_type == LinkPredictionRelation)
		{
			fout.open(logging_base_path + "relation_rank.txt");
		}
		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double rmrr = 0;
		double fmrr = 0;
		double total = data_model.data_test_true.size();
		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++arr_total[data_model.relation_type[i->second]];
		}
		int cnt = 0;
		boost::progress_display cons_bar(data_model.data_test_true.size() / 100);
#pragma omp parallel for
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			pair<pair<int, int>, int> t = *i;
			pair<pair<int, int>, int> tt = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);
			vector<double> pro(data_model.set_entity.size());
			vector<pair<pair<int, int>, int> > triple;
			if (task_type == LinkPredictionRelation || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;
					if (score_i >= prob_triplets(t))
						continue;
					++rmean;
					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
					}
				}
			}
			else
			{
				pro[rmean] = score_i;
				triple.push_back(t);
				string type = data_model.relation_id_to_name[t.second];
				bool flag = true;
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					if (task_type == LinkPredictionHead || part == 1)
					{
						t.first.first = j;
					}
					else
					{
						t.first.second = j;
					}
					//if (type == "_also_see" || type == "_hypernym"|| type=="_member_meronym"|| type=="_hyponym"||type=="_has_part"){
					//	vector<string> head;
					//	vector<string> tail;
					//	boost::split(head, data_model.entity_id_to_name[t.first.first], boost::is_any_of("_"));
					//	boost::split(tail, data_model.entity_id_to_name[t.first.second], boost::is_any_of("_"));
					//	if (head[head.size() - 2] != tail[tail.size() - 2])
					//		flag = false;
					//}
					if (score_i >= prob_triplets(t) || !flag)
						continue;
					++rmean;
					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
						//if (frmean > hit_rank)
						//	break;
						pro[rmean] = prob_triplets(t);
						triple.push_back(t);
					}
				}
			}
#pragma omp critical
			{
				fout << "********************************" << endl;
				++cnt;
				if (cnt % 100 == 0)
				{
					++cons_bar;
				}

				if (frmean < hit_rank)
					++arr_mean[data_model.relation_type[i->second]];

				mean += rmean;//ʵ������
				fmean += frmean;//filter��������
				rmrr += 1.0 / (rmean + 1);//ʵ�ʵĵ���ƽ����
				fmrr += 1.0 / (frmean + 1);//filter���ĵ���ƽ����

				fout << tt.first.first << " " << tt.first.second << " " << tt.second << " " << rmean << " " << frmean << endl;
				int k = 0;
				for (auto ii = triple.begin(); k < rmean&&ii != triple.end(); k++, ii++){
					pair<pair<int, int>, int> ttt = *ii;
					fout << pro[k] << " " << data_model.entity_id_to_name[ttt.first.first] << " " << data_model.relation_id_to_name[ttt.second] << " " << data_model.entity_id_to_name[ttt.first.second] << endl;
				}

				if (rmean < hit_rank)//���м���
					++hits;
				if (frmean < hit_rank)
					++fhits;

			}
		}

		std::cout << endl;
		for (auto i = 1; i <= 4; ++i)//����4�ֲ�ͬ���͹�ϵ��׼ȷ��(filter֮��)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);
		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestMRR = " << rmrr / total << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestMRR = " << rmrr / total;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;

		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestMRR= " << fmrr / total << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestMRR= " << fmrr / total;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;

		std::cout.flush();
	}
	void test_link_prediction_multi_type(int hit_rank = 10, const int part = 0)
	{
		data_model.load_relation_multi_type(data_model.dataset.base_dir + "head_tail_type_train_new.txt");
		//data_model.load_relation_type(data_model.dataset.base_dir + "relation_specific.txt");
		data_model.calc_rel_ent_pos_typescore();
		ofstream fout;
		if (task_type == LinkPredictionHead)
		{
			fout.open(logging_base_path + "head_rank_type.txt");
		}
		if (task_type == LinkPredictionTail)
		{
			fout.open(logging_base_path + "tail_rank.txt");
		}
		if (task_type == LinkPredictionRelation)
		{
			fout.open(logging_base_path + "relation_rank.txt");
		}
		ofstream fout2;
		fout2.open(logging_base_path + "TransT_relation2id.txt");
		for (size_t i = 0; i < data_model.relation_id_to_name.size(); i++)
		{
			fout2 << data_model.relation_id_to_name[i] << "\t" << i << endl;
		}
		fout2.close();
		fout2.open(logging_base_path + "TransT_enttiy2id.txt");
		for (size_t i = 0; i < data_model.entity_id_to_name.size(); i++)
		{
			fout2 << data_model.entity_id_to_name[i] << "\t" << i << endl;
		}
		fout2.close();
		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double rmrr = 0;
		double fmrr = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };


		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++arr_total[data_model.relation_type[i->second]];
		}

		int cnt = 0;

		boost::progress_display cons_bar(data_model.data_test_true.size() / 100);

		map<int, set<int> > ent_type = data_model.type_of_entity;
		map<int, pair<set<int>, set<int> > > rel_type = data_model.type_of_relation;
		int unmatched_head = 0;
		int unmatched_tail = 0;
		set<int> unmatched_head_set;
		set<int> unmatched_tail_set;

#pragma omp parallel for
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			double type_score = 1.0;
			pair<pair<int, int>, int> t = *i;
			pair<pair<int, int>, int> tt = *i;
			vector<double> typeScore(data_model.set_entity.size());
			vector<double> pro(data_model.set_entity.size());
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);
			double t_score_i;

			int relation_id = t.second;
			int entity_id = t.first.first;

			if (task_type == LinkPredictionRelation || part == 2)
			{
				score_i = score_i*data_model.rel_ent_pos_typescore[t.second][t.first.first][0] * data_model.rel_ent_pos_typescore[t.second][t.first.second][1];

				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t)*data_model.rel_ent_pos_typescore[t.second][t.first.first][0] * data_model.rel_ent_pos_typescore[t.second][t.first.second][1])
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;

					}
				}
			}
			else if (task_type == LinkPredictionHead || part == 1){

				typeScore[rmean] = data_model.rel_ent_pos_typescore[t.second][t.first.first][0];
				pro[rmean] = score_i;

				score_i = score_i * data_model.rel_ent_pos_typescore[t.second][t.first.first][0];

				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					t.first.first = j;
					type_score = data_model.rel_ent_pos_typescore[t.second][t.first.first][0];

					//  if (score_i >= prob_triplets(t))

					if (score_i >= prob_triplets(t)*type_score)
						continue;
					++rmean;

					typeScore[rmean] = type_score;
					pro[rmean] = prob_triplets(t);

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
						//if (frmean > hit_rank)
						//	break;
						//temp.push_back(t);
					}
				}
			}
			else{

				typeScore[rmean] = data_model.rel_ent_pos_typescore[t.second][t.first.second][1];
				pro[rmean] = score_i;

				score_i = score_i * data_model.rel_ent_pos_typescore[t.second][t.first.second][1];

				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{

					t.first.second = j;
					type_score = data_model.rel_ent_pos_typescore[t.second][t.first.second][1];

					// if (score_i >= prob_triplets(t))
					if (score_i >= prob_triplets(t)*type_score)
						continue;

					++rmean;
					typeScore[rmean] = type_score;
					pro[rmean] = prob_triplets(t);

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
						//if (frmean > hit_rank)
						//	break;
						//temp.push_back(t);
					}
				}
			}
#pragma omp critical
			{
				fout << "-----------------------" << endl;
				++cnt;
				if (cnt % 100 == 0)
				{
					++cons_bar;
				}

				if (frmean < hit_rank)
					++arr_mean[data_model.relation_type[i->second]];

				mean += rmean;//ʵ������
				fmean += frmean;//filter��������
				rmrr += 1.0 / (rmean + 1);//ʵ�ʵĵ���ƽ����
				fmrr += 1.0 / (frmean + 1);//filter���ĵ���ƽ����

				fout << tt.first.first << " " << tt.first.second << " " << tt.second << " " << rmean << " " << frmean << endl;

				for (int k = 0; k < rmean; k++){

					fout << typeScore[k] << " " << pro[k] * typeScore[k] << endl;
				}

				if (rmean < hit_rank)//���м���
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
			//*/
		}
		if (fout.is_open())
		{
			fout.close();
		}
		std::cout << endl;
		for (auto i = 1; i <= 4; ++i)//����4�ֲ�ͬ���͹�ϵ��׼ȷ��(filter֮��)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "unmatched_head=" << unmatched_head << " unmatched_tail=" << unmatched_tail << endl;
		std::cout << "unmatched_head_set_size=" << unmatched_head_set.size() << " unmatched_tail_set_size=" << unmatched_tail_set.size() << endl;
		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestMRR = " << rmrr / total << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestMRR = " << rmrr / total;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;

		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestMRR= " << fmrr / total << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestMRR= " << fmrr / total;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;

		std::cout.flush();

		cout << cnt << " " << data_model.data_test_true.size() << " " << endl;


	}
	void test(int hit_rank = 10)
	{
		logging.record();
		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;
		test_triplet_classification();
		//return;
		if (task_type == LinkPredictionHead || task_type == LinkPredictionTail || task_type == LinkPredictionRelation)
			test_link_prediction(hit_rank);
		test_triplet_classification();
	}
public: // io functions
	void save(const string& filename)
	{

		double ave_num = 0.0;
		for (auto i = 0; i<count_entity(); ++i)
		{
			ave_num += size_clusters[i];
		}
		cout << "ave_num / count_entity():" << ave_num / count_entity() << endl;
		logging.record() << "ave_num / count_entity():" << ave_num / count_entity();
		ofstream fout(filename, ios::binary);
		for (const Col<double> & ivmatout : embedding_relation)
		{
			fout.write((char*)&ivmatout.n_rows, sizeof(ivmatout.n_rows));
			fout.write((char*)&ivmatout.n_cols, sizeof(ivmatout.n_cols));
			fout.write((char*)ivmatout.memptr(), ivmatout.n_elem * sizeof(double));
		}
		//storage_vmat<double>::save(embedding_relation, fout);
		for (auto &elem_vec : embedding_clusters)
		{
			storage_vmat<double>::save(elem_vec, fout);
		}
		storage_vmat<double>::save(weights_clusters, fout);
		storage_vector<int>::save(size_clusters, fout);
		fout.close();
	}
	void load(const string& filename)
	{
		ifstream fin(filename, ios::binary);
		arma::uword n_size;
		fin.read((char*)&n_size, sizeof(n_size));
		//cout << "n_size=" << n_size << endl;
		embedding_relation.resize(n_size);
		for (Col<double> & it : embedding_relation)
		{
			arma::uword	n_row, n_col;

			fin.read((char*)&n_row, sizeof(n_row));
			fin.read((char*)&n_col, sizeof(n_col));
			it.resize(n_row);
			fin.read((char*)it.memptr(), n_row * n_col * sizeof(double));
		}
		storage_vmat<double>::load(embedding_relation, fin);
		for (auto &elem_vec : embedding_clusters)
		{
			storage_vmat<double>::load(elem_vec, fin);
		}
		storage_vmat<double>::load(weights_clusters, fin);
		storage_vector<int>::load(size_clusters, fin);
		fin.close();
		size_t count = 0;
		for (size_t i = 0; i < embedding_clusters.size(); i++)
		{
			count += embedding_clusters[i].size();
		}
		cout << "average_vec_num=" << ((double)count) / ((double)embedding_clusters.size()) << endl;
	}
};

