package org.firstinspires.ftc.teamcode;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.openftc.easyopencv.OpenCvCamera;
import org.openftc.easyopencv.OpenCvCameraFactory;
import org.openftc.easyopencv.OpenCvCameraRotation;
import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.List;
import java.util.ArrayList;


public class BlueCubeDetectionPipeline extends OpenCvPipeline {
    Mat hsv = new Mat();
    Mat mask = new Mat();
    Mat morphed = new Mat();

    @Override
    public Mat processFrame(Mat input) {

        // Convert the image from RGB to HSV
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

        // Define the range for blue color
        Scalar lowerBlue = new Scalar(90, 100, 100);
        Scalar upperBlue = new Scalar(140, 255, 255);

        // Threshold the image to get only blue colors
        Core.inRange(hsv, lowerBlue, upperBlue, mask);

        // Use morphological operations to remove noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(mask, morphed, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(morphed, morphed, Imgproc.MORPH_CLOSE, kernel);

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(morphed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            double aspectRatio = (double) rect.width / rect.height;
            Imgproc.rectangle(input, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
            /* 
            if (aspectRatio > 0.8 && aspectRatio < 1.2 && rect.area() > 1000) { // Assuming cube contours will be roughly square
                Imgproc.rectangle(input, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
            }*/
        }

        return input;
    }
}
