
#include "TransT.h"

// 400s for each experiment.
int main(int argc, char* argv[])
{
	//Dataset FB15K("FB15K", "K:\\ARB\\database\\KGE\\FB15k\\", "train.txt", "valid_with_neg.txt", "test_with_neg.txt", "entity2type.txt", false);
	//Dataset WN18("WN18", "K:\\ARB\\database\\KGE\\WN18\\", "wn_train_new.txt", "wn_valid_new.txt", "wn_test_new.txt", true);
	//Dataset FB15K("FB15K", "D:\\shh\\Embedding-master\\FB15K\\", "train.txt", "dev.txt", "test.txt", false);
	Dataset FB15K("FB15K", "D:\\shh\\transt\\FB15K\\", "train.txt", "valid_with_neg.txt", "test_with_neg.txt", "entity2type.txt", false);
	printf("dataset init finish!\n");
	const string path = "D:\\shh\\transt\\result\\";

	int dim = 200;
	double alpha = 0.001;
	double training_threshold = 3.0;
	int n_cluster = 1;
	double CRP_factor = 0.001;
	int step_before = 20;
	int max_cluster_size = 20;
	bool sot = false;
	bool be_weight_normalized = true;
	int max_epos = 1000;
	printf("create transt\n");
	TransT * model = new TransT(FB15K, LinkPredictionHead, path, dim, alpha, training_threshold, n_cluster, max_cluster_size, CRP_factor, step_before, sot, be_weight_normalized);
	//model = new TransT_type_single(FB15K, LinkPredictionHead, path, dim, alpha, training_threshold, n_cluster, max_cluster_size, CRP_factor, step_before, variance_bound, sot, be_weight_normalized);
	printf("start to train\n");
	string modelname = typeid(*model).name();
	string time = "-0421_1";

	model->run(max_epos);
	model->save(path + modelname.substr(6, modelname.length() - 6) + "-"
		+ to_string(dim) + "-"
		+ to_string(max_epos) + "-"
		+ to_string(CRP_factor) + "-"
		+ to_string(step_before) + "-"
		+ to_string(alpha) + "-"
		+ to_string(training_threshold) + "-"
		+ to_string(be_weight_normalized) + time + ".model");

	//model->test();

	//   
	//model->load(path + modelname.substr(6,modelname.length()-6) + "-"
	//	+ to_string(dim) + "-"
	//	+ to_string(max_epos) + "-"
	//	+ to_string(CRP_factor) + "-"
	//	+ to_string(step_before) + "-"
	//	+ to_string(alpha) + "-"
	//	+ to_string(training_threshold) + "-"
	//	+ to_string(be_weight_normalized) + time + ".model");

	//	//
	//model->report(path + modelname.substr(6, modelname.length() - 6) + "-"
	//	+ to_string(dim) + "-"
	//	+ to_string(max_epos) + "-"
	//	+ to_string(CRP_factor) + "-"
	//	+ to_string(step_before) + "-"
	//	+ to_string(alpha) + "-"
	//	+ to_string(training_threshold) + "-"
	//	+ to_string(be_weight_normalized) + time + ".txt");

	printf("start to test\n");

	model->changeTaskType(LinkPredictionHead);
	model->test();

	model->changeTaskType(LinkPredictionTail);
	model->test();

	model->changeTaskType(LinkPredictionRelation);
	model->test(1);


	delete model;
	return 0;
}
