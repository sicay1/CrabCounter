#include "ParticleCounter.h"
#include "TGMTfile.h"
#include <TGMTimage.h>
#include <math.h>
#include "TGMTcontour.h"
#include "TGMTConfig.h"
#include "TGMTdebugger.h"
#include "TGMTutil.h"
#include "TGMTdraw.h"

ParticleCounter* ParticleCounter::instance = nullptr;
#define INI_SECTION "ParticleCounter"

////////////////////////////////////////////////////////////////////////////////////////////////////

ParticleCounter::ParticleCounter()
{
	m_minSize = 2;
	m_maxSize = 10;

	m_minSize = GetTGMTConfig()->ReadValueInt(INI_SECTION, "min_size", m_minSize);
	m_maxSize = GetTGMTConfig()->ReadValueInt(INI_SECTION, "max_size", m_maxSize);
	m_estimateSize = GetTGMTConfig()->ReadValueInt(INI_SECTION, "estimate_size", m_minSize);
	ASSERT(m_minSize <= m_estimateSize && m_estimateSize <= m_maxSize, "Estimate size must bigger min size and smaller max size");
	m_estimateSize = CorrectSize(m_estimateSize);

	PrintMessageYellow("Count crab in size [%d; %d]", m_minSize, m_maxSize);


	m_positiveColor = GetTGMTConfig()->ReadValueBool("ParticleCounter", "positive_color");

	std::string strInputSize = GetTGMTConfig()->ReadValueString("ParticleCounter", "resize_input");
	if (TGMTutil::SplitString(strInputSize, ',').size() == 2)
	{
		auto split = TGMTutil::SplitString(strInputSize, ',');
		m_inputSize.width = Str2Int(split[0]);
		m_inputSize.height = Str2Int(split[1]);
	}

	m_drawSolid = GetTGMTConfig()->ReadValueBool("ParticleCounter", "draw_solid") ? -1 : 1;

	int color = GetTGMTConfig()->ReadValueInt("ParticleCounter", "draw_color");
	switch (color)
	{
	case 1:
		m_color = GREEN;
		break;
	case 2:
		m_color = BLUE;
		break;
	case 3:
		m_color = WHITE;
		break;
	case 4:
		m_color = BLACK;
		break;
	case 5:
		m_color = YELLOW;
		break;
	case 6:
		m_color = RED;
		break;
	default:
		m_color = UNDEFINED_COLOR;
		break;
	}


	//m_thresh = CorrectSize(m_thresh);
	m_debug = GetTGMTConfig()->ReadValueBool("ParticleCounter", "debug");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ParticleCounter::~ParticleCounter()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<cv::Point>> ClampContourByArea(std::vector<std::vector<cv::Point>> contours, int minValue, int maxValue)
{
	std::vector<std::vector<cv::Point>> result;
	for (int i = 0; i < contours.size(); ++i)
	{
		cv::Rect r = cv::boundingRect(contours[i]);

		if (minValue <= r.area() && r.area() <= maxValue)
		{
			result.push_back(contours[i]);
		}
	}
	return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<cv::Point>> ClampContourBySize(std::vector<std::vector<cv::Point>> contours, int minValue, int maxValue)
{
	std::vector<std::vector<cv::Point>> result;
	for (int i = 0; i < contours.size(); ++i)
	{
		cv::Rect r = cv::boundingRect(contours[i]);
		float distance = abs(r.width - r.height) / (r.width + r.height);
		if (distance < 0.5)
		{
			if (minValue <= r.width &&r.width <= maxValue  && minValue <= r.height && r.height <= maxValue)
			{
				result.push_back(contours[i]);
			}
		}
		
	}
	return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int ParticleCounter::CountParticle(cv::Mat &matInput)
{
	if (m_inputSize.area() > 0)
	{
		cv::resize(matInput, matInput, m_inputSize);
	}
	cv::Mat matGray = TGMTimage::ConvertToGray(matInput);

	
	if(m_debug)
		cv::imshow("mat gray", matGray);

	//blur
	cv::Mat matBlur;
	cv::GaussianBlur(matGray, matBlur, cv::Size(m_estimateSize, m_estimateSize), 0, 0);

	if (!m_positiveColor)
		cv::bitwise_not(matBlur, matBlur);

	//threshold
	cv::Mat matBinary;
	cv::adaptiveThreshold(matBlur, matBinary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 21, 0);
	if (m_debug)
		cv::imshow("binary", matBinary);


	//Finding sure foreground area
	cv::Mat dist_transform;
	cv::distanceTransform(matBinary, dist_transform, CV_DIST_L2, 3);


	cv::Mat matNorm;
	cv::normalize(dist_transform, matNorm, 0, 1, cv::NORM_MINMAX);
	if (m_debug)
		cv::imshow("matNorm", matNorm);


	double min, max;
	cv::minMaxLoc(dist_transform, &min, &max);
	cv::Mat sure_fg;
	cv::threshold(dist_transform, sure_fg, 0.35*max, 255, CV_THRESH_BINARY);
	if (m_debug)
		cv::imshow("sure_fg", sure_fg);


	//convert to 1 channel
	sure_fg.convertTo(sure_fg, CV_8U);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(sure_fg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	contours = ClampContourBySize(contours, m_minSize, m_maxSize);

	if (matInput.channels() == 1)
	{
		cv::cvtColor(matInput, matInput, CV_GRAY2BGR);
	}
	
	TGMTcontour::DrawContours(matInput, contours, m_color, m_drawSolid);
	TGMTdraw::PutText(matInput, cv::Point(0, 30), RED, "%d", contours.size());
	cv::imshow("Output", matInput);

	return contours.size();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int ParticleCounter::CountParticle(std::string imgPath)
{
	cv::Mat mat = cv::imread(imgPath);
	return CountParticle(mat);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int ParticleCounter::CorrectSize(int size)
{
	if (size % 2 == 0)
		size++;
	if (size < 5)
		size = 5;

	return size;
}