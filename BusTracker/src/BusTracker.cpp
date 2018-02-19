#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <map>
#include <set>
#include <vector>
 
using namespace cv;
using namespace std;

#define VIDEO_FILENAME "../data/taipei-hires-2017-04-08.mp4"
#define BOX_FILENAME "../data/taipei-hires-2017-04-08.csv"
#define OUTPUT_FILENAME "../output/boxes.mp4"

#define TOTAL_FRAMES 18000

#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720

#define FPS 30

void readStartingBoxes(map<int, Rect2d>& boxes) {
	set<int> used;

	ifstream file(BOX_FILENAME);
	string line;
	string value;

	getline(file, line, '\n');
	int frame = 0;
	while (frame <= TOTAL_FRAMES) {
		getline(file, line, '\n');
		stringstream ss(line);

		getline(ss, value, ',');
		frame = stoi(value);

		getline(ss, value, ',');
		if (value == "bus") {
			getline(ss, value, ',');
			float confidence = stof(value);

			getline(ss, value, ',');
			int xmin = (int) stof(value);

			getline(ss, value, ',');
			int ymin = (int) stof(value);

			getline(ss, value, ',');
			int xmax = (int) stof(value);

			getline(ss, value, ',');
			int ymax = (int) stof(value);

			getline(ss, value, ',');
			int index = stoi(value);

			if (frame <= TOTAL_FRAMES && used.find(index) == used.end()) {
				Rect2d bbox(xmin, ymin, xmax - xmin, ymax - ymin);
				boxes[frame] = bbox;
				used.insert(index);
			}
		}
	}
}
 
int main(int argc, char **argv)
{
	VideoCapture video(VIDEO_FILENAME);
	VideoWriter videoWriter(OUTPUT_FILENAME, CV_FOURCC('H', '2', '6', '4'), FPS, Size(FRAME_WIDTH, FRAME_HEIGHT));

	map<int, Rect2d> startingBoxes;
	readStartingBoxes(startingBoxes);

	int frameCount = 0;
	Mat frame;
	vector<Rect2d> bboxes;
	vector<Ptr<Tracker> > trackers;
	while (video.read(frame)) {
		if (frameCount % 100 == 0) {
			cout << "Processing frame " << frameCount << endl;
		}
		// If the current frame has a bounding box, initializes new tracker
		if (startingBoxes.find(frameCount) != startingBoxes.end()) {
			Rect2d bbox = startingBoxes[frameCount];
			Ptr<Tracker> tracker = TrackerKCF::create();
			tracker->init(frame, bbox);
			trackers.push_back(tracker);
			bboxes.push_back(bbox);
		}
		// Updates all current trackers
		for (int i = 0; i < trackers.size(); i ++) {
			bool success = trackers[i]->update(frame, bboxes[i]);
			rectangle(frame, bboxes[i], Scalar(255, 0, 0), 2, 1);
			Rect2d inside = (bboxes[i] & Rect2d(0, 0, frame.cols, frame.rows));
			// Deletes trackers that are no longer tracking anything
			if (!success || inside.width == 0 || inside.height == 0) {
				trackers[i].release();
				trackers.erase(trackers.begin() + i);
				bboxes.erase(bboxes.begin() + i);
				i --;
			}
		}
		videoWriter.write(frame);
		frameCount ++;
	}
	video.release();
	videoWriter.release();
	return 0;
}
