#pragma once
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>
import std;

class PCA {
private:
	Eigen::MatrixXd data_;  //原始数据
	Eigen::MatrixXd mean_;  //均值
	Eigen::MatrixXd components_;  //主成分
	Eigen::VectorXd explained_variance_;  //解释方差--每个主成分方向上的方差大小（特征值）
	Eigen::VectorXd explained_variance_ratio_;  //方差解释率--每个主成分解释了多少比例的总方差

public:
	PCA() {}

	void fit(const Eigen::MatrixXd& data, int n_components = -1) { //--拟合PCA模型
		data_ = data;

		//1.中心化数据
		int n_samples = data.rows();
		int n_features = data.cols();

		//计算均值
		mean_ = data.colwise().mean();

		//中心化
		Eigen::MatrixXd centered = data.rowwise() - mean_.transpose(); //rowwise--按行操作，transpose--转置

		//计算协方差矩阵--可视化每种特征与其他特征的联系
		Eigen::MatrixXd covariance = (centered.adjoint() * centered) / double(n_samples - 1);

		//计算特征值和特征向量
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance);

		//获取特征值和特征向量（升序）
		Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
		Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors();

		//按特征值降序排序
		std::vector<std::pair<double, int>> sorted_indices;
		for (int i = 0; i < eigenvalues.size(); ++i) {
			sorted_indices.push_back({ eigenvalues[i],i });
		}
		std::sort(sorted_indices.begin(), sorted_indices.end(),
			[](const auto& a, const auto& b) {return a.first > b.first; });

		//确定主成分数量
		if (n_components <= 0 || n_components > n_features) {
			n_components = n_features;
		}

		//选择前n_components个主成分
		components_.resize(n_features, n_components);
		explained_variance_.resize(n_components);

		for (int i = 0; i < n_components; ++i) {
			int idx = sorted_indices[i].second;
			components_.col(i) = eigenvectors.col(idx);
			explained_variance_[i] = eigenvalues[idx];
		}

		//计算方差解释率
		double total_variance = eigenvalues.sum();
		explained_variance_ratio_ = explained_variance_ / total_variance;
	}

	//数据变换（降维）
	Eigen::MatrixXd transform(const Eigen::MatrixXd& data)const {
		Eigen::MatrixXd centered = data.rowwise() - mean_.transpose();
		return centered * components_;
	}

	//逆变换（重建数据）
	Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& transformed)const {
		return transformed * components_ + mean_.transpose();
	}

	Eigen::MatrixXd getComponents() const { return components_; }
	Eigen::MatrixXd getMean() const { return mean_; }
	Eigen::VectorXd getExplainedVariance() const { return explained_variance_; }
	Eigen::VectorXd getExplainedVarianceRatio() const { return explained_variance_ratio_; }

	Eigen::VectorXd getCumulativeExplainedVarianceRatio() const {
		Eigen::VectorXd cumulative(explained_variance_ratio_.size());
		double sum = 0;
		for (int i = 0; i < explained_variance_ratio_.size(); ++i) {
			sum += explained_variance_ratio_[i];
			cumulative[i] = sum;
		}
		return cumulative;
	}
};