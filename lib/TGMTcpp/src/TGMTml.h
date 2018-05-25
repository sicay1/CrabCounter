#pragma once
#include "stdafx.h"
#include <opencv2\ml\ml.hpp>


#define HORIZONTAL	1
#define VERTICAL    0

using namespace cv::ml;

//#define GetTGMTml TGMTml::GetInstance

class TGMTml
{
	enum DataType
	{
		raw,
		trained
	}m_dataType;
	
	void DrawVisualFeatures(cv::Mat character, cv::Mat hhist, cv::Mat vhist, cv::Mat lowData);
	cv::Mat ProjectedHistogram(cv::Mat img, int t);
	cv::Mat GetVisualHistogram(cv::Mat hist, int type);
	cv::Mat GetProjectedHistogramMat(cv::Mat input);
protected:

	cv::Size m_desireSize;
	cv::Mat m_matData;
	cv::Mat m_matLabel;

	bool m_enableThreshold;
	int m_thresholdValue;
	bool m_enableProjectedHistogram;

	
	void TrainData(std::vector<std::string> imgPaths, std::vector<int> labels);
	cv::Mat PrepareMatData(cv::Mat matInput);
public:
	TGMTml();
	~TGMTml();	

	bool TrainData(std::string dirPath);	
	virtual bool TrainData(cv::Mat matData, cv::Mat matLabel) = 0;

	bool LoadData(std::string fileName);
	void SaveData(std::string fileName);

	//load file mat in xml file
	void SaveMatData(std::string fileName);
	bool LoadMatData(std::string fileName);

	virtual bool LoadModel(std::string modelFile) = 0;
	virtual void SaveModel(std::string modelFile) = 0;

	virtual float Predict(cv::Mat matInput) = 0;
	virtual float Predict(std::string imgPath) = 0;
};

