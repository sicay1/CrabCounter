#pragma once
#include "stdafx.h"

#define GetCrabCounter CrabCounter::GetInstance

class CrabCounter
{
	static CrabCounter* instance;

	cv::Scalar m_color;
	int m_drawSolid;
	int m_estimateSize;
	int m_minSize;
	int m_maxSize;
	bool m_positiveColor;
	bool m_debug;


	cv::Size m_inputSize;

	enum CrabType
	{
		Pepper,
		Melon
	}m_crabType;

	int CountPepperCrab(std::string imgPath);
	int CountPepperCrab(cv::Mat& matInput);
	int CountMelonCrab(std::string imgPath);
	int CountMelonCrab(cv::Mat& matInput);

	int CorrectSize(int size);
public:
	CrabCounter();
	~CrabCounter();

	static CrabCounter* GetInstance()
	{
		if (!instance)
			instance = new CrabCounter();
		return instance;
	}

	

	int CountCrab(std::string imgPath);
	int CountCrab(cv::Mat& matInput);
};
