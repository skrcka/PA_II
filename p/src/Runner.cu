#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cudaDefs.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace cv;


// CUDA kernel for license plate detection
__global__ void detectLicensePlateKernel(const unsigned char* image, int width, int height, unsigned char* output, int otsuThreshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * width + x;

	if (x < width && y < height)
	{
		// Apply a simple edge detection filter (Sobel)
		int gx = 0, gy = 0;

		if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
		{
			int topLeft = image[(y - 1) * width + (x - 1)];
			int top = image[(y - 1) * width + x];
			int topRight = image[(y - 1) * width + (x + 1)];
			int left = image[y * width + (x - 1)];
			int right = image[y * width + (x + 1)];
			int bottomLeft = image[(y + 1) * width + (x - 1)];
			int bottom = image[(y + 1) * width + x];
			int bottomRight = image[(y + 1) * width + (x + 1)];
			gx = -topLeft - 2 * left - bottomLeft + topRight + 2 * right + bottomRight;
			gy = -topLeft - 2 * top - topRight + bottomLeft + 2 * bottom + bottomRight;
		}

		int edgeMagnitude = min(255, max(0, abs(gx) + abs(gy)));

		// Apply otsu thresholding
		output[index] = (edgeMagnitude > otsuThreshold) ? 255 : 0;
	}

	__syncthreads();
	// Perform morphological operations (dilation) to connect potential license plate characters
	if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
	{
		int dilationSum = output[(y - 1) * width + x] + output[(y + 1) * width + x] + output[y * width + (x - 1)] + output[y * width + (x + 1)];
		if (dilationSum > 0)
		{
			output[index] = 255;
		}
	}
}

int computeOtsuThreshold(const Mat& grayImage) {
	int histogram[256] = { 0 };
	int totalPixels = grayImage.rows * grayImage.cols;

	// Calculate the histogram
	for (int y = 0; y < grayImage.rows; ++y) {
		for (int x = 0; x < grayImage.cols; ++x) {
			int intensity = grayImage.at<uchar>(y, x);
			histogram[intensity]++;
		}
	}

	// Calculate the Otsu's threshold value
	float sum = 0;
	for (int i = 0; i < 256; ++i) {
		sum += i * histogram[i];
	}

	float sumB = 0;
	int wB = 0;
	int wF = 0;
	float maxVariance = 0;
	int threshold = 0;

	for (int i = 0; i < 256; ++i) {
		wB += histogram[i]; // Weight of the background
		if (wB == 0) continue;
		wF = totalPixels - wB; // Weight of the foreground
		if (wF == 0) break;

		sumB += static_cast<float>(i * histogram[i]);

		float mB = sumB / wB; // Mean of the background
		float mF = (sum - sumB) / wF; // Mean of the foreground

		// Calculate Between Class Variance
		float variance = static_cast<float>(wB) * wF * (mB - mF) * (mB - mF);

		if (variance > maxVariance) {
			maxVariance = variance;
			threshold = i;
		}
	}

	return threshold;
}

void detectLicensePlate(const Mat& inputImage, Mat& outputImage)
{
	int width = inputImage.cols;
	int height = inputImage.rows;

	// Convert the input image to grayscale
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
	outputImage = grayImage.clone();
	int otsuThreshold = computeOtsuThreshold(grayImage);

	// Allocate GPU memory for the input and output image
	unsigned char* d_inputImage;
	unsigned char* d_outputImage;
	size_t image_size = width * height * sizeof(unsigned char);
	cudaMalloc((void**)&d_inputImage, image_size);
	cudaMalloc((void**)&d_outputImage, image_size);

	// Copy the grayscale image data to the GPU
	cudaMemcpy(d_inputImage, grayImage.data, image_size, cudaMemcpyHostToDevice);
	// Configure CUDA grid and block dimensions
	dim3 blockDim(16, 16);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

	// Launch the CUDA kernel for license plate detection
	detectLicensePlateKernel << <gridDim, blockDim >> > (d_inputImage, width, height, d_outputImage, otsuThreshold);

	// Wait for the kernel to finish and check for errors
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cerr << "Error in detectLicensePlateKernel: " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}

	// Copy the processed image data back to the host
	cudaMemcpy(outputImage.data, d_outputImage, image_size, cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_inputImage);
	cudaFree(d_outputImage);
}

int main(int argc, char* argv[])
{
	// Load the input image
	Mat inputImage = imread("C:\\Users\\GAMES\\Downloads\\xd2.png", IMREAD_COLOR);
	if (inputImage.empty())
	{
		cerr << "Error: Could not open or find the image." << endl;
		return EXIT_FAILURE;
	}

	// Detect license plate in the input image
	Mat outputImage;
	detectLicensePlate(inputImage, outputImage);

	// Display the detected license plate image
	imshow("Contours", outputImage);
	waitKey(0);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(outputImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> filteredContours;

	for (const auto& contour : contours)
	{
		Rect rect = boundingRect(contour);
		double aspectRatio = static_cast<double>(rect.width) / rect.height;
		double area = contourArea(contour);

		// Adjust the conditions below based on the specific characteristics of license plates in your region
		if (aspectRatio > 2 && aspectRatio < 6 && area > 1000 && area < 10000)
		{
			filteredContours.push_back(contour);
		}
	}

	Rect licensePlateBoundingBox;
	double maxArea = 0.0;

	for (const auto& contour : filteredContours)
	{
		Rect rect = boundingRect(contour);
		double area = contourArea(contour);

		if (area > maxArea)
		{
			maxArea = area;
			licensePlateBoundingBox = rect;
		}
	}

	Scalar boundingBoxColor = Scalar(0, 255, 0); // Green color
	int thickness = 2;
	rectangle(inputImage, licensePlateBoundingBox, boundingBoxColor, thickness);

	imshow("Detected License Plate Bounding Box", inputImage);
	waitKey(0);

	return EXIT_SUCCESS;
}
