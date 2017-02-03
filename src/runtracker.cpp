/// @desc A wrapper to run KCF
///
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>

#include "kcftracker.hpp"

#include <dirent.h>

void DisplayUsage();

void GetFilesInDirectory(const std::string directory_absolute_path, 
			 std::vector<std::string> &out);

using namespace std;
using namespace cv;
using namespace boost::filesystem;

int main(int argc, char* argv[]){
  if (argc > 5 || argc == 1){
    DisplayUsage();
    return -1;
  }
  
  //default setting
  bool HOG = true;
  bool FIXEDWINDOW = false;
  bool MULTISCALE = true;
  bool SILENT = true;
  bool LAB = false;
  std::string data_directory_path;
  
  for(int i = 0; i < argc; i++){
    if ( strcmp (argv[i], "data") == 0 )
      data_directory_path = argv[++i];
    //break;
    if ( strcmp (argv[i], "hog") == 0 )
      HOG = true;
    if ( strcmp (argv[i], "fixed_window") == 0 )
      FIXEDWINDOW = true;
    if ( strcmp (argv[i], "singlescale") == 0 )
      MULTISCALE = false;
    if ( strcmp (argv[i], "show") == 0 )
      SILENT = false;
    if ( strcmp (argv[i], "lab") == 0 ){
      LAB = true;
      HOG = true;
    }
    if ( strcmp (argv[i], "gray") == 0 )
      HOG = false;
  }

  std::cout << "[info] data directory = " << data_directory_path << std::endl;
	
  // Create KCFTracker object
  KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

  // Image frame to read
  Mat frame;

  // Tracker results, <x, y, w, h>
  Rect result;

  // Read the names of image files
  std::vector<std::string> image_file_names;
  GetFilesInDirectory(data_directory_path+"img/", image_file_names);
  std::cout << "[info] Number of input images=" << image_file_names.size() << std::endl;

  // Need to sort. Otherwise the images do not read in the
  // chronological order.
  sort(image_file_names.begin(), image_file_names.end());

  // Read groundtruth for the 1st frame
  ifstream groundtruthFile;
  string groundtruth = "groundtruth_rect.txt";
  groundtruthFile.open(data_directory_path+groundtruth);
  string firstLine;
  getline(groundtruthFile, firstLine);
  groundtruthFile.close();
  istringstream ss(firstLine);

  // Read groundtruth like a dumb
  float x1, y1, x2, y2, x3, y3, x4, y4;
  char ch;
  ss >> x1;
  ss >> ch;
  ss >> y1;
  ss >> ch;
  ss >> x2;
  ss >> ch;
  ss >> y2;
  ss >> ch;
  ss >> x3;
  ss >> ch;
  ss >> y3;
  ss >> ch;
  ss >> x4;
  ss >> ch;
  ss >> y4; 

  // Using min and max of X and Y for groundtruth rectangle
  float xMin =  min(x1, min(x2, min(x3, x4)));
  float yMin =  min(y1, min(y2, min(y3, y4)));
  float width = max(x1, max(x2, max(x3, x4))) - xMin;
  float height = max(y1, max(y2, max(y3, y4))) - yMin;

  // Write Results
  ofstream resultsFile;
  string resultsPath = "output.txt";
  resultsFile.open(resultsPath);
  
  // Frame counter
  int nFrames = 0;

  clock_t begin = clock();
  auto start = std::chrono::high_resolution_clock::now();

  // output box property
  cv::Scalar output_box_color(0, 0, 255);
  int box_thickness = 2;

  namedWindow("KCF", WINDOW_NORMAL);
  resizeWindow("KCF", 1600, 900);
  
  for( std::vector<std::string>::const_iterator itr = image_file_names.begin();
       itr != image_file_names.end(); ++itr ) {

    std::cout << "[info] processing (" << (nFrames+1)
	      << "/" << image_file_names.size() << "), "
	      << (*itr) << std::endl;

    // read image
    frame = imread((*itr), CV_LOAD_IMAGE_COLOR);

    // If it's the first frame, give the groundtruth to the tracker
    // for initialization
    if (nFrames == 0) {
      tracker.init( Rect(xMin, yMin, width, height), frame );
      rectangle( frame, 
		 Point( xMin, yMin ), 
		 Point( xMin+width, yMin+height), 
		 output_box_color, box_thickness, 8 );
      resultsFile << xMin << "," << yMin << "," << width << "," << height << std::endl;
    } else { // otherwise, update
      result = tracker.update(frame);
      rectangle( frame, 
		 Point( result.x, result.y ), 
		 Point( result.x+result.width, result.y+result.height), 
		 output_box_color, box_thickness, 8 );
      resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
    }
      
    nFrames++;

    //if(!SILENT){
      imshow("KCF", frame);
      waitKey(1);
      //}
  }

  // Measuring the elapsed time
  double elapsed_time = double(clock() - begin) / CLOCKS_PER_SEC;
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  
  std::cout << "[info] Elapsed time (CPU time)=" << elapsed_time 
	    << ", number of images=" << image_file_names.size()
	    << ", elapsed time per frame in sec=" << elapsed_time / image_file_names.size() << std::endl;

  std::cout << "[info] Elapsed time (Wall clock)=" << elapsed.count()
	    << ", number of images=" << image_file_names.size()
	    << ", elapsed time per frame in sec=" << elapsed_time / image_file_names.size() << std::endl;
  
  resultsFile.close();
}

void DisplayUsage()
{
  std::cout << "[Usage] kcf [option] " << std::endl;
  std::cout << "- option: " << std::endl;
  std::cout << " -- default: Use a default setting (hog=true, fixedwindow=false, multiscale=true, slient=true, lab=false)" << std::endl;
  std::cout << " -- hog: Use the HOG feature." << std::endl;
  std::cout << " -- fixedwindow: fix window size, otherwise use ROI size." << std::endl;
  std::cout << " -- multiscale: " << std::endl;
  std::cout << " -- show: " << std::endl;
  std::cout << " -- lab: Use Lab color. " << std::endl;
  std::cout << " -- gray: Use the raw pixels as feature." << std::endl;
}

void GetFilesInDirectory(const string directory_absolute_path,
			 std::vector<std::string> &filename_list)
{
  path p(directory_absolute_path);

  for (auto itr = directory_iterator(p); 
       itr != directory_iterator(); ++itr)
  {
    // if the entry is not a directory
    if ( !is_directory(itr->path()) )
    {
      std::string file_name = directory_absolute_path + 
	itr->path().filename().string();
      filename_list.push_back(file_name);
    } else {
      continue;
    }
  }
}
