#include <vector>

#include "readnetpbm.h"
#include "perceptron.h"

int main(int argc, char* argv[]) {
	srand((unsigned)time(0));

	std::vector<sample> trainset;

	trainset.push_back(sample("Bobsled"));
	read_netpbm_bw("./imgsrc/s1.ppm", trainset.back().get_pvalues());

	trainset.push_back(sample("Slide"));
	read_netpbm_bw("./imgsrc/s2.ppm", trainset.back().get_pvalues());

	trainset.push_back(sample("Skates"));
	read_netpbm_bw("./imgsrc/s3.ppm", trainset.back().get_pvalues());

	trainset.push_back(sample("Fight"));
	read_netpbm_bw("./imgsrc/s4.ppm", trainset.back().get_pvalues());

	perceptron3<uint8_t> net(6, 8, 4);
	net.init_neurons(trainset.back().get_size(), 6, 8);
	net.init_trainset(&trainset);
	net.teach_1e(&trainset);

	return 0;
}