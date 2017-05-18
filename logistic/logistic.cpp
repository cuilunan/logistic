#include <iostream>
#include <cstdio>
#include <String>
#include <cstdlib>
#include <vector>
#include <map>
#include <cmath>
using namespace std;

/////训练数据节点
class trainVectorNode {
private:
	int index;
	double attributeValue;
public:
	trainVectorNode::trainVectorNode() :index(0), attributeValue(0.0) {}
	int getIndex();
	double getAttributeValue();
	void setIndex(int index);
	void setAttributeValue(double value);
};
int trainVectorNode::getIndex() {
	return this->index;
}
double trainVectorNode::getAttributeValue() {
	return this->attributeValue;
}
void trainVectorNode::setIndex(int index) {
	this->index = index;
}
void trainVectorNode::setAttributeValue(double value) {

	this->attributeValue = value;
}


///////////神经元节点
class neuroNode {
public:
	double sumOfInput(vector<double> W,vector<double> X);
	double activate_Sigmod(double x);
};
double neuroNode::sumOfInput(vector<double> W, vector<double> X) {
	double sum = 0;
	for (int i = 0;i<W.size();i++) {
		sum += W[i] * X[i];
	}
	return sum;
}

double neuroNode::activate_Sigmod(double x) {
	return 1 / (double)(1 + exp(-x));

}

///梯度下降更新权值
void updateW_Gradscent(vector<double> &W, vector<double> X,int label,double learning_rate) {
	neuroNode ne;
	double predict = ne.activate_Sigmod(ne.sumOfInput(W, X));
	//if (label - predict < 0.5 || predict - label < 0.5)
		//return;
	double punish = 10;
	for (int i = 0; i < W.size(); i++) {
		W[i] =W[i]*(1-learning_rate*(punish/(double)W.size()))+ learning_rate*X[i] * (label - predict);
	}
}


void train(vector<double> &W, vector<vector<trainVectorNode>> &trainData, vector<int> &label) {
	///迭代j次
	vector<vector<double>> X;
	for (vector<trainVectorNode> train_vector : trainData) {
		vector<double> x(W.size(), 0);
		for (trainVectorNode node : train_vector) {
			x[node.getIndex() - 1] = node.getAttributeValue();
		}
		X.push_back(x);
	}

	for (int j = 0; j < 1; j++) {
		int i = 0;
		for (vector<trainVectorNode> train_vector : trainData) {
			updateW_Gradscent(W, X[i], label[i], 0.001);
			i++;
			///
			cout << i << endl;
		}

	}
}

void readFile(vector<int> &label,vector<vector<trainVectorNode>> &trainData,string fileName) {
	char s[1024];
	FILE *fp = fopen(fileName.c_str(), "r");
	fgets(s,1000,fp);
	fflush(fp);
	while (!feof(fp)) {
		int classfication;
		if (fscanf(fp, "%d", &classfication) == EOF) {
			break;
		}
		if (classfication == -1) {
			label.push_back(0);
		}
		else {
			label.push_back(1);
		}
		vector<trainVectorNode> ve;
		while (1) {
			int att;
			double value;
			if (fgetc(fp)!='\n') {
				fscanf(fp, "%d:%lf", &att, &value);
				fflush(fp);
				trainVectorNode *node = new trainVectorNode();
				node->setIndex(att);
				node->setAttributeValue(value);
				ve.push_back(*node);
				free(node);
			}
			else {
				
				trainData.push_back(ve);
				break;
			}

		}
		

	}
	fclose(fp);

}

void readTestFile(vector<int> &test_label, vector<vector<trainVectorNode>> &testData, string fileName) {
	char s[1024];
	FILE *fp = fopen(fileName.c_str(), "r");
	fgets(s, 1000, fp);
	fflush(fp);
	while (!feof(fp)) {
		int classfication;
		if (fscanf(fp, "%d", &classfication) == EOF) {
			break;
		}
		test_label.push_back(classfication);
		vector<trainVectorNode> ve;
		while (1) {
			int att;
			double value;
			if (fgetc(fp) != '\n') {
				fscanf(fp, "%d:%lf", &att, &value);
				fflush(fp);
				trainVectorNode *node = new trainVectorNode();
				node->setIndex(att);
				node->setAttributeValue(value);
				ve.push_back(*node);
				free(node);
			}
			else {

				testData.push_back(ve);
				break;
			}

		}


	}
	fclose(fp);
}

double accurracy(vector<double> W, vector<vector<trainVectorNode>> testData, vector<int> test_label) {
	double accuracy = 0.0;
	int num = 0;
	neuroNode n;
	int i = 0;
	for (vector<trainVectorNode> ve: testData) {
		vector<double> X(W.size(), 0);
		for (trainVectorNode node : ve) {
			X[node.getIndex() - 1] = node.getAttributeValue();
		}
		double result = n.activate_Sigmod(n.sumOfInput(W, X));
		cout << test_label[i] << "==>" << result << endl;
		if (result >= 0.5) {
			if (test_label[i] == 1) {
				num++;
			}
		}
		else {
			if (test_label[i] == -1) {
				num++;
			}
		}
		i++;
	}
	accuracy = num /(double) testData.size();
	return accuracy;
}

void main() {
	vector<int> label;
	vector<vector<trainVectorNode>> trainData;
	vector<int> test_label;
	vector<vector<trainVectorNode>> testData;
	string fileName = "E:\\hit_laboratory\\c++\\logistic-regression-sgd-master\\train.dat";
	readFile(label, trainData, fileName);
	int max = 0;
	for (vector<trainVectorNode> re : trainData) {
		for (trainVectorNode node : re) {
			if (node.getIndex() > max) {
				max = node.getIndex();
			}
		}
	}
	vector<double> W(max, 0);
	train(W, trainData, label);
	readTestFile(test_label, testData,"E:\\hit_laboratory\\c++\\logistic-regression-sgd-master\\test.dat");
	double accuracy=accurracy(W, testData, test_label);
	cout << "accuracy:"<<accuracy << endl;
	system("pause");
}