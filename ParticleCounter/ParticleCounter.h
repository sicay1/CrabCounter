#pragma once
#include "stdafx.h"

#define GetParticleCounter ParticleCounter::GetInstance

class ParticleCounter
{
	static ParticleCounter* instance;

	cv::Scalar m_color;
	int m_drawSolid;
	int m_estimateSize;
	int m_minSize;
	int m_maxSize;
	bool m_positiveColor;
	bool m_debug;


	cv::Size m_inputSize;

	
	int CorrectSize(int size);
public:
	ParticleCounter();
	~ParticleCounter();

	static ParticleCounter* GetInstance()
	{
		if (!instance)
			instance = new ParticleCounter();
		return instance;
	}

	
	int CountParticle(cv::Mat& matInput);
	int CountParticle(std::string imgPath);
};
