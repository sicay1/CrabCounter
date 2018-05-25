// ParticleCounter.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ParticleCounter.h"
#include <tchar.h>
#include "TGMTdraw.h"
#include "TGMTdebugger.h"
#include "TGMTfile.h"
#include "TGMTConfig.h"
#include "TGMTutil.h"

/////////////////////////////////////////////////////////////////////////////

void _tmain(int argc, _TCHAR* argv[])
{
	GetTGMTConfig()->LoadSettingFromFile();

	std::string dir = GetTGMTConfig()->ReadValueString("ParticleCounter", "dir");
	if (dir.empty())
		dir = TGMTfile::GetCurrentDir();

	PrintMessage("Read image in dir: %s", dir.c_str());

	std::vector<std::string> images = TGMTfile::GetImageFilesInDir(dir);
	if (images.size() == 0)
	{
		PrintError("Can not load any image in dir: %s", dir.c_str());
		return;
	}
	else
	{
		PrintMessage("Loaded %d image", images.size());
	}

	TGMTfile::CreateDir("output");
	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat mat = cv::imread(images[i]);
		int count = GetParticleCounter()->CountParticle(mat);
		std::cout << images[i].c_str() << ": " << count << "\n";
		
		WriteImageAsync(mat, "output\\%s_detected.jpg", TGMTfile::GetFileNameWithoutExtension(images[i]).c_str());
	}

	cv::waitKey();
	getchar();
}

