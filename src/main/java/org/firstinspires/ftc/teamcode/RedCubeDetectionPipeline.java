package org.firstinspires.ftc.teamcode;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.core.*;
import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

public class RedCubeDetectionPipeline extends OpenCvPipeline {
    private Telemetry telemetry;  // Add Telemetry object
    Mat hsv = new Mat();
    Mat mask = new Mat();
    Mat mask1 = new Mat(); // Added for the lower red range
    Mat mask2 = new Mat(); // Added for the upper red range
    Mat morphed = new Mat();

    // Create rectangle zones for the red cubes
    public float side_width = 240;
    public float side_height = 312;
    public float center_width = 300;
    public float center_height = 330;

    // Set center coordinates for the red cubes
    public float left_x = 23;
    public float left_y = 91;

    public float center_x = 267;
    public float center_y = 62;

    public float right_x = 572;
    public float right_y = 100;

    // Adjust the HSV range for red
    public Scalar lowerRed1 = new Scalar(0, 100, 24);
    public Scalar upperRed1 = new Scalar(10, 255, 255);
    public Scalar lowerRed2 = new Scalar(160, 100, 100);
    public Scalar upperRed2 = new Scalar(180, 255, 255);

    public RedCubeDetectionPipeline(Telemetry telemetry) {  // Constructor to initialize telemetry
        this.telemetry = telemetry;
    }

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        Core.inRange(hsv, lowerRed1, upperRed1, mask1);
        Core.inRange(hsv, lowerRed2, upperRed2, mask2);
        Core.bitwise_or(mask1, mask2, mask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(mask, morphed, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(morphed, morphed, Imgproc.MORPH_CLOSE, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(morphed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Draw rectangles around the zones in green
        Imgproc.rectangle(input, new Point(center_x, center_y), new Point(center_x + center_width, center_y + center_height), new Scalar(0, 255, 0), 1);
        Imgproc.rectangle(input, new Point(left_x, left_y), new Point(left_x + side_width, left_y + side_height), new Scalar(0,255, 0), 1);
        Imgproc.rectangle(input, new Point(right_x, right_y), new Point(right_x + side_width, right_y + side_height), new Scalar(0,255, 0), 1);

        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            double centerX = rect.x + rect.width / 2.0;
            double centerY = rect.y + rect.height / 2.0;

            double area = Imgproc.contourArea(contour);

            // detect location 
            if (area < 1000) {
                break;
            }

            if (centerX > center_x && centerX < center_x + center_width && centerY > center_y && centerY < center_y + center_height) {
                telemetry.addData("Location", "Center");
            } else if (centerX > left_x && centerX < left_x + side_width && centerY > left_y && centerY < left_y + side_height) {
                telemetry.addData("Location", "Left");
            } else if (centerX > right_x && centerX < right_x + side_width && centerY > right_y && centerY < right_y + side_height) {
                telemetry.addData("Location", "Right");
            } else {
                telemetry.addData("Location", "Unknown");
            }

            // Telemetry data for the area of each red object
            telemetry.addData("Area", area);

            // Telemetry data for the center coordinates of each red object
            telemetry.addData("CenterX, CenterY", centerX + ", " + centerY);  

            Imgproc.rectangle(input, rect.tl(), rect.br(), new Scalar(0, 0, 255), 2);  // Red color rectangle
        }

        telemetry.update();  // Ensure you update telemetry after adding data
        return input;
    }
}
