#include "TGMTml.h"
#include "TGMTConfig.h"
#include "TGMTdebugger.h"
#include "TGMTfile.h"
#include "TGMTutil.h"
#include "TGMTimage.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TGMTml::TGMTml()
{
	m_enableProjectedHistogram = GetTGMTConfig()->ReadValueBool("TGMTml", "enable_projected_histogram");
	std::string desireSize = GetTGMTConfig()->ReadValueString("TGMTml", "desire_size");
	auto splitDesireSize = TGMTutil::SplitString(desireSize, ',');
	if (splitDesireSize.size() == 2)
	{
		m_desireSize = cv::Size(Str2Int(splitDesireSize[0]), Str2Int(splitDesireSize[1]));
	}

	int type = GetTGMTConfig()->ReadValueInt("TGMTml", "data_type");
	if (type >= 0 && type < sizeof(DataType))
	{
		m_dataType = (DataType)type;
	}
	else
	{
		PrintError("DataType is invalid");
	}


	m_enableThreshold = GetTGMTConfig()->ReadValueBool("TGMTml", "enable_threshold");
	if (m_enableThreshold)
	{
		m_thresholdValue = GetTGMTConfig()->ReadValueInt("TGMTml", "threshold_value");
		if (m_thresholdValue <= 0 || m_thresholdValue > 255)
		{
			m_enableThreshold = false;
		}
	}

}

////////////////////////////////////////////////////////////////////////////////////////////////////

TGMTml::~TGMTml()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat TGMTml::GetProjectedHistogramMat(cv::Mat in)
{
	//Histogram features
	cv::Mat vhist = ProjectedHistogram(in, VERTICAL);
	cv::Mat hhist = ProjectedHistogram(in, HORIZONTAL);

	//Low data feature
	cv::Mat lowData;
	cv::resize(in, lowData, m_desireSize);

#if _DEBUG
	DrawVisualFeatures(in, hhist, vhist, lowData);
#endif

	//Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.rows;

	cv::Mat out = cv::Mat::zeros(1, numCols, CV_32F);
	//Asign values to feature
	int j = 0;
	for (int i = 0; i < vhist.cols; i++)
	{
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}

	for (int i = 0; i < hhist.cols; i++)
	{
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}

	for (int x = 0; x < lowData.cols; x++)
	{
		for (int y = 0; y < lowData.rows; y++)
		{
			out.at<float>(j) = (float)lowData.at<uchar>( y, x);
			j++;
		}
	}
	return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TGMTml::DrawVisualFeatures(cv::Mat character, cv::Mat hhist, cv::Mat vhist, cv::Mat lowData)
{
	cv::Mat img(character.rows + 101, character.cols + 101, CV_8UC3, BLACK);
	cv::Mat ch;
	cv::Mat ld;

	

	cv::Mat roi = img(cv::Rect(0, 101, character.cols, character.rows));
	cv::cvtColor(character, ch, CV_GRAY2RGB);	
	ch.copyTo(roi);


	roi = img(cv::Rect(character.cols + 1, 101, 100, character.rows));
	cv::Mat hh = GetVisualHistogram(hhist, HORIZONTAL);
	hh.copyTo(roi);

	roi = img(cv::Rect(0, 0, character.cols, 100));
	cv::Mat hv = GetVisualHistogram(vhist, VERTICAL);
	hv.copyTo(roi);

	roi = img(cv::Rect(character.cols + 1, 0, 100, 100));
	cv::resize(lowData, ld, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
	cv::cvtColor(ld, ld, CV_GRAY2RGB);
	ld.copyTo(roi);

	cv::line(img, cv::Point(0, 100), cv::Point(img.cols, 100), RED);
	cv::line(img, cv::Point(character.cols, 0), cv::Point(character.cols, img.rows), RED);

	cv::imshow("Visual Features", img);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat TGMTml::GetVisualHistogram(cv::Mat hist, int type)
{
	int size = 100;
	cv::Mat imHist;


	if (type == HORIZONTAL)
	{
		imHist.create(cv::Size(size, hist.cols), CV_8UC3);
	}
	else
	{
		imHist.create(cv::Size(hist.cols, size), CV_8UC3);
	}

	imHist = cv::Scalar(55, 55, 55);

	for (int i = 0; i < hist.cols; i++)
	{
		float value = hist.at<float>(i);
		int maxval = (int)(value*size);

		cv::Point pt1;
		cv::Point pt2, pt3, pt4;

		if (type == HORIZONTAL)
		{
			pt1.x = pt3.x = 0;
			pt2.x = pt4.x = maxval;
			pt1.y = pt2.y = i;
			pt3.y = pt4.y = i + 1;

			cv::line(imHist, pt1, pt2, cv::Scalar(220, 220, 220), 1, 8, 0);
			cv::line(imHist, pt3, pt4, cv::Scalar(34, 34, 34), 1, 8, 0);

			pt3.y = pt4.y = i + 2;
			cv::line(imHist, pt3, pt4, cv::Scalar(44, 44, 44), 1, 8, 0);
			pt3.y = pt4.y = i + 3;
			cv::line(imHist, pt3, pt4, cv::Scalar(50, 50, 50), 1, 8, 0);
		}
		else
		{
			pt1.x = pt2.x = i;
			pt3.x = pt4.x = i + 1;
			pt1.y = pt3.y = 100;
			pt2.y = pt4.y = 100 - maxval;


			cv::line(imHist, pt1, pt2, cv::Scalar(220, 220, 220), 1, 8, 0);
			cv::line(imHist, pt3, pt4, cv::Scalar(34, 34, 34), 1, 8, 0);

			pt3.x = pt4.x = i + 2;
			cv::line(imHist, pt3, pt4, cv::Scalar(44, 44, 44), 1, 8, 0);
			pt3.x = pt4.x = i + 3;
			cv::line(imHist, pt3, pt4, cv::Scalar(50, 50, 50), 1, 8, 0);

		}
	}

	return imHist;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat TGMTml::ProjectedHistogram(cv::Mat img, int orient)
{
	int size = orient ? img.rows : img.cols;
	cv::Mat mhist = cv::Mat::zeros(1, size, CV_32F);

	for (int j = 0; j < size; j++)
	{
		cv::Mat data = orient ? img.row(j) : img.col(j);
		mhist.at<float>(j) = cv::countNonZero(data);
	}

	//Normalize histogram
	double min, max;
	cv::minMaxLoc(mhist, &min, &max);

	if (max > 0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);

	return mhist;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TGMTml::TrainData(std::string dirPath)
{
	std::vector<std::string> files = TGMTfile::GetImageFilesInDir(dirPath, true);
	if (files.size() == 0)
	{
		PrintError("Not found any image");
		return false;
	}

	std::vector<std::string> validFiles;
	std::vector<int> validLabels;
	for (int i = 0; i < files.size(); i++)
	{
		std::string filePath = files[i];
		std::string parentDir = TGMTfile::GetParentDir(filePath, false);
		if (parentDir.length() > 2)
			continue;

		validFiles.push_back(filePath);
		int label = (char)parentDir[0];
		validLabels.push_back(label);
	}

	TrainData(validFiles, validLabels);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TGMTml::TrainData(std::vector<std::string> imgPaths, std::vector<int> labels)
{
	if (imgPaths.size() == 0 || labels.size() == 0)
	{
		PrintError("Data input is empty");
		return;
	}
	if (imgPaths.size() != labels.size())
	{
		PrintError("Image set not equal label set");
		return;
	}
	START_COUNT_TIME("train_knn");
	SET_CONSOLE_TITLE("Traning knn data...");
	size_t numMats = imgPaths.size();

	int matArea = m_desireSize.width + m_desireSize.height + m_desireSize.width * m_desireSize.height;
	m_matData = cv::Mat(numMats, matArea, CV_32FC1);
	m_matLabel = cv::Mat(numMats, 1, CV_32SC1);

	//prepare train set
	for (size_t fileIndex = 0; fileIndex < numMats; fileIndex++)
	{

		//set label
		m_matLabel.at<int>(fileIndex, 0) = labels[fileIndex];

		cv::Mat mat = cv::imread(imgPaths[fileIndex], CV_LOAD_IMAGE_GRAYSCALE);
		mat = PrepareMatData(mat);
		
		mat.row(0).copyTo(m_matData.row(fileIndex));
	}

	TrainData(m_matData, m_matLabel);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TGMTml::SaveMatData(std::string fileName)
{
	cv::FileStorage file(fileName, cv::FileStorage::WRITE);
	file << "traindata" << m_matData;
	file << "trainlabel" << m_matLabel;
	file.release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TGMTml::LoadData(std::string fileName)
{
	if (!TGMTfile::FileExist(fileName))
	{
		PrintError("File does not exist: %s", fileName.c_str());
		return false;
	}

	switch (m_dataType)
	{
	case DataType::raw:
		return LoadMatData(fileName);
	case DataType::trained:
		return LoadModel(fileName);
	};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TGMTml::SaveData(std::string fileName)
{
	switch (m_dataType)
	{
	case DataType::raw:
		SaveMatData(fileName);
		break;
	case DataType::trained:
		SaveModel(fileName);
		break;
	};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TGMTml::LoadMatData(std::string trainDataFile)
{
	cv::FileStorage fs;
	fs.open(trainDataFile, cv::FileStorage::READ);
	ASSERT(fs.isOpened(), "Can not load %s", trainDataFile.c_str());
	cv::Mat matData;
	cv::Mat matClass;
	fs["traindata"] >> matData;
	fs["trainlabel"] >> matClass;
	fs.release();

	if (!matData.data)
	{
		PrintError("Mat data is empty");
		return false;
	}
	if (!matClass.data)
	{
		PrintError("Mat label is empty");
		return false;
	}

	return TrainData(matData, matClass);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat TGMTml::PrepareMatData(cv::Mat matInput)
{
	ASSERT(matInput.data, "Mat input is null");
	cv::Mat mat = matInput.clone();
	mat = TGMTimage::ConvertToGray(mat);

	if (mat.cols != m_desireSize.width || mat.rows != m_desireSize.height)
	{
		cv::resize(mat, mat, cv::Size(m_desireSize.width, m_desireSize.height));
	}

	if (m_enableThreshold)
	{
		cv::threshold(mat, mat, m_thresholdValue, 255, CV_THRESH_BINARY);
	}

	if (m_enableProjectedHistogram)
		mat = GetProjectedHistogramMat(mat);
	else
		mat = mat.reshape(1, 1);

	mat.convertTo(mat, CV_32FC1);

	return mat;
}